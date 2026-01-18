from typing import Collection, Dict, Iterator, List, Optional, Set, Tuple, cast
import torch
from torch import Tensor
from torchmetrics.utilities import rank_zero_warn
Compute the final panoptic quality from interim values.

    Args:
        iou_sum: the iou sum from the update step
        true_positives: the TP value from the update step
        false_positives: the FP value from the update step
        false_negatives: the FN value from the update step

    Returns:
        Panoptic quality as a tensor containing a single scalar.

    