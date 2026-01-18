import math
import numpy as np
import mxnet as mx
from ..context import current_context
from ..random import uniform
from ..base import _as_list
from . import ndarray
def _get_rescale_grad(rescale_grad, ctx=mx.cpu()):
    if not isinstance(rescale_grad, ndarray.NDArray):
        return ndarray.full(shape=(1,), val=rescale_grad, ctx=ctx)
    else:
        return rescale_grad.as_in_context(ctx)