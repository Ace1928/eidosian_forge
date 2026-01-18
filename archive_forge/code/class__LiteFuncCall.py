import collections as _collections
import copy as _copy
import json as _json
import uuid as _uuid
from tensorflow.core.framework import attr_value_pb2 as _attr_value_pb2
from tensorflow.core.framework import graph_pb2 as _graph_pb2
from tensorflow.core.framework import node_def_pb2 as _node_def_pb2
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.python.framework import ops as _ops
from tensorflow.python.framework import tensor_util as _tensor_util
from tensorflow.python.framework.graph_util_impl import _bfs_for_reachable_nodes
from tensorflow.python.framework.graph_util_impl import _extract_graph_summary
from tensorflow.python.ops import array_ops as _array_ops
from tensorflow.python.util import compat as _compat
from tensorflow.python.util import deprecation as _deprecation
from tensorflow.python.util.all_util import remove_undocumented
from tensorflow.python.util.tf_export import tf_export as _tf_export
class _LiteFuncCall:
    """Represent a TensorFlow Lite custom function.

  This is uses to accumulate found hints in the graphdef into a single
  conceptual unit.

  Attributes:
    inputs: inputs to the op (hash from index # to argument)
    outputs: outputs to the op (hash from index # to argument)
    function_name: the tflite custom op name to use
    uuid: a unique call id for this particular call  (i.e. multiple function
      calls would have the same function_name but different uuids.
    params: A param name to key value for op constant data. I.e. for axis on a
      reduction, strides on a convolution, etc.
    level: Level of the OpHint.
    children_inputs_mappings: If the Ophint has children, children inputs
      mappings indicate how their inputs & outputs are mapped.
  """

    def __init__(self):
        self.inputs = {}
        self.outputs = {}
        self.function_name = None
        self.uuid = None
        self.params = {}
        self.level = -1
        self.children_inputs_mappings = {}

    def flattened_inputs_and_outputs(self):
        """Return a list of inputs and outputs in a flattened format.

    Returns:
      Tuple of (inputs, outputs). where input and output i a list of names.
    """

        def _flatten(input_or_output_dict):
            flattened_items = []
            for item in input_or_output_dict.values():
                flattened_items.extend(item.flatten())
            return flattened_items
        return (_flatten(self.inputs), _flatten(self.outputs))

    def __str__(self):

        def format_args(items):
            s = ''
            for idx, item in items.iteritems():
                s += '\t\t%d:\n' % idx + str(item)
            return s
        inputs_str = '\tInputs\n' + format_args(self.inputs)
        outputs_str = '\tOutputs\n' + format_args(self.outputs)
        return 'tflite function %s call %s level %d \n\tinputs:\n\t\t%s\n\toutputs:\n\t\t%s' % (self.function_name, self.uuid, self.level, inputs_str, outputs_str)