import math
import numpy as np
import mxnet as mx
from ..context import current_context
from ..random import uniform
from ..base import _as_list
from . import ndarray
Converts "inputs", possibly typed mxnet NDArray, a numpy ndarray, other python types,
        to the given type
        