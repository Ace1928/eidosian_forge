from typing import Optional, Tuple
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.utilities.prints import rank_zero_warn
def _unable_to_use_bias_correction_warning(metric_name: str) -> None:
    rank_zero_warn(f'Unable to compute {metric_name} using bias correction. Please consider to set `bias_correction=False`.')