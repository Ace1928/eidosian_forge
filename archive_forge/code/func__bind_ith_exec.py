import logging
from collections import OrderedDict
from .. import context as ctx
from .. import ndarray as nd
from ..io import DataDesc
from ..executor_manager import _split_input_slice
from ..ndarray import _DTYPE_MX_TO_NP
def _bind_ith_exec(self, i, data_shapes, label_shapes, shared_group):
    """Internal utility function to bind the i-th executor.
        This function utilizes simple_bind python interface.
        """
    shared_exec = None if shared_group is None else shared_group.execs[i]
    context = self.contexts[i]
    shared_data_arrays = self.shared_data_arrays[i]
    input_shapes = dict(data_shapes)
    if label_shapes is not None:
        input_shapes.update(dict(label_shapes))
    input_types = {x.name: x.dtype for x in data_shapes}
    attr_dict = self.symbol.attr_dict()
    for sym_name in self.symbol.list_inputs():
        if sym_name in input_types and sym_name in attr_dict and ('__dtype__' in attr_dict[sym_name]) and (attr_dict[sym_name]['__dtype__'] != '-1'):
            input_types[sym_name] = _DTYPE_MX_TO_NP[int(attr_dict[sym_name]['__dtype__'])]
    if label_shapes is not None:
        input_types.update({x.name: x.dtype for x in label_shapes})
    group2ctx = self.group2ctxs[i]
    executor = self.symbol.simple_bind(ctx=context, grad_req=self.grad_req, type_dict=input_types, shared_arg_names=self.param_names, shared_exec=shared_exec, group2ctx=group2ctx, shared_buffer=shared_data_arrays, **input_shapes)
    self._total_exec_bytes += int(executor.debug_str().split('\n')[-3].split()[1])
    return executor