from typing import Optional, Tuple
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.utilities.prints import rank_zero_warn
def _nominal_input_validation(nan_strategy: str, nan_replace_value: Optional[float]) -> None:
    if nan_strategy not in ['replace', 'drop']:
        raise ValueError(f"Argument `nan_strategy` is expected to be one of `['replace', 'drop']`, but got {nan_strategy}")
    if nan_strategy == 'replace' and (not isinstance(nan_replace_value, (float, int))):
        raise ValueError(f"Argument `nan_replace` is expected to be of a type `int` or `float` when `nan_strategy = 'replace`, but got {nan_replace_value}")