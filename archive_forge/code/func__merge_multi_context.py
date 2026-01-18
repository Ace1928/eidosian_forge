import logging
from collections import OrderedDict
from .. import context as ctx
from .. import ndarray as nd
from ..io import DataDesc
from ..executor_manager import _split_input_slice
from ..ndarray import _DTYPE_MX_TO_NP
def _merge_multi_context(outputs, major_axis):
    """Merge outputs that lives on multiple context into one, so that they look
    like living on one context.
    """
    rets = []
    for tensors, axis in zip(outputs, major_axis):
        if axis >= 0:
            if len(tensors) == 1:
                rets.append(tensors[0])
            else:
                rets.append(nd.concat(*[tensor.as_in_context(tensors[0].context) for tensor in tensors], dim=axis))
        else:
            rets.append(tensors[0])
    return rets