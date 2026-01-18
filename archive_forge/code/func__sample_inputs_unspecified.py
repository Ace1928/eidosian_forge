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
def _sample_inputs_unspecified(self, *args, **kwargs):
    """Raises an NotImplemented exception in a OpInfo instance creation
        that specifies supports_sparse(|_csr|_csc|_bsr|_bsc)=True
        without specifying the corresponding sample function as
        sample_inputs_sparse_(coo|csr|csc|bsr|bsc)_func.

        To avoid this, either define the corresponding sample function,
        or re-map unsupported samples to error inputs in an appropiate

          opinfo/definitions/sparse.py:_validate_sample_input_sparse_<op>

        function.
        """
    raise NotImplementedError('no sample function specified')