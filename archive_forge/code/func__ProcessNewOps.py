import contextlib
from tensorflow.core.framework import graph_pb2
from tensorflow.python import tf2
from tensorflow.python.client import pywrap_tf_session as c_api
from tensorflow.python.framework import c_api_util
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import errors
from tensorflow.python.framework import function
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import control_flow_util
from tensorflow.python.util import compat
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.tf_export import tf_export
def _ProcessNewOps(graph):
    """Processes the newly-added TF_Operations in `graph`."""
    colocation_pairs = {}
    for new_op in graph._add_new_tf_operations(compute_devices=False):
        original_device = new_op.device
        new_op._set_device('')
        colocation_names = _GetColocationNames(new_op)
        if colocation_names:
            colocation_pairs[new_op] = colocation_names
        else:
            with _MaybeDevice(original_device):
                graph._apply_device_functions(new_op)
    for op, coloc_op_list in colocation_pairs.items():
        coloc_device = None
        for coloc_op_name in coloc_op_list:
            try:
                coloc_op = graph._get_operation_by_name(coloc_op_name)
            except KeyError:
                if tf2.enabled() or control_flow_util.EnableControlFlowV2(graph):
                    continue
                raise ValueError(f'Specified colocation to an op: {coloc_op_name} that does not exist during import for op: {op.name}')
            if coloc_op.device:
                coloc_device = pydev.DeviceSpec.from_string(coloc_op.device)
                break
        if coloc_device:
            op._set_device(coloc_device)