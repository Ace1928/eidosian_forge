from numbers import Integral, Real
import numpy as np
from scipy import linalg
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from ..base import (
from ..exceptions import NotFittedError
from ..metrics.pairwise import pairwise_kernels
from ..preprocessing import KernelCenterer
from ..utils._arpack import _init_arpack_v0
from ..utils._param_validation import Interval, StrOptions
from ..utils.extmath import _randomized_eigsh, svd_flip
from ..utils.validation import (
Number of transformed output features.