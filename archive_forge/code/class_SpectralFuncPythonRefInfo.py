import unittest
from functools import partial
from typing import List
import numpy as np
import torch
from torch.testing import make_tensor
from torch.testing._internal.common_cuda import SM53OrLater
from torch.testing._internal.common_device_type import precisionOverride
from torch.testing._internal.common_dtype import (
from torch.testing._internal.common_utils import TEST_SCIPY, TEST_WITH_ROCM
from torch.testing._internal.opinfo.core import (
from torch.testing._internal.opinfo.refs import (
class SpectralFuncPythonRefInfo(SpectralFuncInfo):
    """
    An OpInfo for a Python reference of an elementwise unary operation.
    """

    def __init__(self, name, *, op=None, torch_opinfo_name, torch_opinfo_variant='', **kwargs):
        self.torch_opinfo_name = torch_opinfo_name
        self.torch_opinfo = _find_referenced_opinfo(torch_opinfo_name, torch_opinfo_variant, op_db=op_db)
        assert isinstance(self.torch_opinfo, SpectralFuncInfo)
        inherited = self.torch_opinfo._original_spectral_func_args
        ukwargs = _inherit_constructor_args(name, op, inherited, kwargs)
        super().__init__(**ukwargs)