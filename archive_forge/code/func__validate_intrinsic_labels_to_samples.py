from typing import Optional, Union
import torch
from torch import Tensor, tensor
from typing_extensions import Literal
from torchmetrics.utilities.checks import _check_same_shape
def _validate_intrinsic_labels_to_samples(num_labels: int, num_samples: int) -> None:
    """Validate that the number of labels are in the correct range."""
    if not 1 < num_labels < num_samples:
        raise ValueError(f'Number of detected clusters must be greater than one and less than the number of samples.Got {num_labels} clusters and {num_samples} samples.')