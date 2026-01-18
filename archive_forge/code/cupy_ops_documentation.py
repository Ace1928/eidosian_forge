import numpy
from .. import registry
from ..compat import cublas, cupy, cupyx
from ..types import DeviceTypes
from ..util import (
from . import _custom_kernels
from .numpy_ops import NumpyOps
from .ops import Ops
Given an (M, N) sequence of vectors, return an (M, N*(nW*2+1)) sequence.
        The new sequence is constructed by concatenating nW preceding and succeeding
        vectors onto each column in the sequence, to extract a window of features.
        