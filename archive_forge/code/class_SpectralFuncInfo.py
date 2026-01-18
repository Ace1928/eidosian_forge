import collections
import collections.abc
import math
import operator
import unittest
from dataclasses import asdict, dataclass
from enum import Enum
from functools import partial
from itertools import product
from typing import Any, Callable, Iterable, List, Optional, Tuple
from torchgen.utils import dataclass_repr
import torch
from torch.testing import make_tensor
from torch.testing._internal.common_device_type import (
from torch.testing._internal.common_dtype import (
from torch.testing._internal.common_utils import (
from torch.testing._internal.opinfo import utils
class SpectralFuncInfo(OpInfo):
    """Operator information for torch.fft transforms."""

    def __init__(self, name, *, ref=None, dtypes=floating_and_complex_types(), ndimensional: SpectralFuncType, sample_inputs_func=sample_inputs_spectral_ops, decorators=None, **kwargs):
        self._original_spectral_func_args = dict(locals()).copy()
        self._original_spectral_func_args.update(kwargs)
        decorators = list(decorators) if decorators is not None else []
        decorators += [skipCPUIfNoFFT, DecorateInfo(toleranceOverride({torch.chalf: tol(0.04, 0.04)}), 'TestCommon', 'test_complex_half_reference_testing')]
        super().__init__(name=name, dtypes=dtypes, decorators=decorators, sample_inputs_func=sample_inputs_func, **kwargs)
        self.ref = ref
        self.ndimensional = ndimensional