from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.regression.utils import _check_data_shape_to_num_outputs
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.data import _bincount, _cumsum, dim_zero_cat
from torchmetrics.utilities.enums import EnumStr
class _MetricVariant(EnumStr):
    """Enumerate for metric variants."""
    A = 'a'
    B = 'b'
    C = 'c'

    @staticmethod
    def _name() -> str:
        return 'variant'