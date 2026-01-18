import warnings
from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.precision_recall_curve import (
from torchmetrics.functional.classification.roc import (
from torchmetrics.utilities.enums import ClassificationTask
def _convert_fpr_to_specificity(fpr: Tensor) -> Tensor:
    """Convert  fprs to specificity."""
    return 1 - fpr