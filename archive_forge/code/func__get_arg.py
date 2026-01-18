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
def _get_arg(self, name, unpack):
    assert name in self._required_arg_names
    if name not in self._arg_cache:
        fn_name = name + '_fn'
        size_name = name + '_size'
        if name in self._extra_kwargs:
            self._arg_cache[name] = self._extra_kwargs[name]
        elif fn_name in self._extra_kwargs:
            self._arg_cache[name] = self._extra_kwargs[fn_name]()
        else:
            assert size_name in self._extra_kwargs, f'Missing `{name}`, `{size_name}` or `{fn_name}` for {self.get_name()}'

            def map_tensor_sizes(sizes):
                if isinstance(sizes, list):
                    return [map_tensor_sizes(s) for s in sizes]
                elif isinstance(sizes, torch.Tensor):
                    return sizes.double()
                else:
                    return torch.randn(sizes)
            self._arg_cache[name] = map_tensor_sizes(self._extra_kwargs[size_name])
    return self._unpack(self._arg_cache[name]) if unpack else self._arg_cache[name]