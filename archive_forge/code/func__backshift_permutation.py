from functools import partial
from typing import List, Optional, Tuple, Union
import torch
import torch._prims as prims
import torch._prims_common as utils
import torch._refs as refs
import torch._refs.linalg as linalg
from torch import Tensor
from torch._prims_common import (
from torch._prims_common.wrappers import (
from torch._decomp import register_decomposition
def _backshift_permutation(dim0, dim1, ndim):
    ret = [i for i in range(ndim) if i != dim0 and i != dim1]
    ret.extend((dim0, dim1))
    return ret