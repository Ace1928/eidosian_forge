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
class ShapeFuncInfo(OpInfo):
    """Early version of a specialized OpInfo for Shape manipulating operations like tile and roll"""

    def __init__(self, name, *, ref, dtypes=floating_types(), dtypesIfCUDA=None, dtypesIfROCM=None, sample_inputs_func=None, **kwargs):
        super().__init__(name, dtypes=dtypes, dtypesIfCUDA=dtypesIfCUDA, dtypesIfROCM=dtypesIfROCM, sample_inputs_func=sample_inputs_func, **kwargs)
        self.ref = ref