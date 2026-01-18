import logging
from collections import OrderedDict
from .. import context as ctx
from .. import ndarray as nd
from ..io import DataDesc
from ..executor_manager import _split_input_slice
from ..ndarray import _DTYPE_MX_TO_NP
def _collect_arrays(self):
    """Collect internal arrays from executors."""
    if self.slices:
        self.data_arrays = [[(self.slices[i], e.arg_dict[name]) for i, e in enumerate(self.execs)] for name, _ in self.data_shapes]
    else:
        self.data_arrays = [[(slice(i, i + 1), e.arg_dict[name]) for i, e in enumerate(self.execs)] for name, _ in self.data_shapes]
    self.state_arrays = [[e.arg_dict[name] for e in self.execs] for name in self.state_names]
    if self.label_shapes is not None:
        self.label_arrays = [[(self.slices[i], e.arg_dict[name]) for i, e in enumerate(self.execs)] for name, _ in self.label_shapes]
    else:
        self.label_arrays = None
    self.param_arrays = [[exec_.arg_arrays[i] for exec_ in self.execs] for i, name in enumerate(self.arg_names) if name in self.param_names]
    if self.for_training:
        self.grad_arrays = [[exec_.grad_arrays[i] for exec_ in self.execs] for i, name in enumerate(self.arg_names) if name in self.param_names]
    else:
        self.grad_arrays = None
    data_names = [x[0] for x in self.data_shapes]
    if self.inputs_need_grad:
        self.input_grad_arrays = [[exec_.grad_arrays[self.arg_names.index(name)] for exec_ in self.execs] for name in data_names if name in self.arg_names]
    else:
        self.input_grad_arrays = None
    self.aux_arrays = [[exec_.aux_arrays[i] for exec_ in self.execs] for i in range(len(self.aux_names))]