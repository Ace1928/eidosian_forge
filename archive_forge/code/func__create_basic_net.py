from abc import abstractmethod
import tempfile
import unittest
from copy import deepcopy
from functools import reduce, partial, wraps
from itertools import product
from operator import mul
from math import pi
import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import _reduction as _Reduction
from torch.testing._internal.common_utils import TestCase, to_gpu, freeze_rng_state, is_iterable, \
from torch.testing._internal.common_cuda import TEST_CUDA, SM90OrLater
from torch.autograd.gradcheck import _get_numerical_jacobian, _iter_tensors
from torch.autograd import Variable
from torch.types import _TensorOrTensors
import torch.backends.cudnn
from typing import Dict, Callable, Tuple, List, Sequence, Union, Any
def _create_basic_net():

    class Layer(nn.Module):

        def __init__(self):
            super().__init__()
            self.layer_dummy_param = nn.Parameter(torch.empty(3, 5))
            self.register_buffer('layer_dummy_buf', torch.zeros(1, 3, 3, 7))

    class Net(nn.Module):

        def __init__(self):
            super().__init__()
            self.l1 = Layer()
            self.dummy_param = nn.Parameter(torch.empty(3, 5))
            self.register_buffer('dummy_buf', torch.zeros(7, 3, 3, 1))
    l = Layer()
    n = Net()
    s = nn.Sequential(n, n)
    return (l, n, s)