from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.regression.utils import _check_data_shape_to_num_outputs
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.data import _bincount, _cumsum, dim_zero_cat
from torchmetrics.utilities.enums import EnumStr
class _TestAlternative(EnumStr):
    """Enumerate for test alternative options."""
    TWO_SIDED = 'two-sided'
    LESS = 'less'
    GREATER = 'greater'

    @staticmethod
    def _name() -> str:
        return 'alternative'