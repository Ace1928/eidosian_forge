from typing import Optional, Tuple, Union
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.confusion_matrix import (
from torchmetrics.utilities.enums import ClassificationTaskNoMultilabel
def _ce_compute(confidences: Tensor, accuracies: Tensor, bin_boundaries: Union[Tensor, int], norm: str='l1', debias: bool=False) -> Tensor:
    """Compute the calibration error given the provided bin boundaries and norm.

    Args:
        confidences: The confidence (i.e. predicted prob) of the top1 prediction.
        accuracies: 1.0 if the top-1 prediction was correct, 0.0 otherwise.
        bin_boundaries: Bin boundaries separating the ``linspace`` from 0 to 1.
        norm: Norm function to use when computing calibration error. Defaults to "l1".
        debias: Apply debiasing to L2 norm computation as in
            `Verified Uncertainty Calibration`_. Defaults to False.

    Raises:
        ValueError: If an unsupported norm function is provided.

    Returns:
        Tensor: Calibration error scalar.

    """
    if isinstance(bin_boundaries, int):
        bin_boundaries = torch.linspace(0, 1, bin_boundaries + 1, dtype=confidences.dtype, device=confidences.device)
    if norm not in {'l1', 'l2', 'max'}:
        raise ValueError(f"Argument `norm` is expected to be one of 'l1', 'l2', 'max' but got {norm}")
    with torch.no_grad():
        acc_bin, conf_bin, prop_bin = _binning_bucketize(confidences, accuracies, bin_boundaries)
    if norm == 'l1':
        return torch.sum(torch.abs(acc_bin - conf_bin) * prop_bin)
    if norm == 'max':
        ce = torch.max(torch.abs(acc_bin - conf_bin))
    if norm == 'l2':
        ce = torch.sum(torch.pow(acc_bin - conf_bin, 2) * prop_bin)
        if debias:
            debias_bins = acc_bin * (acc_bin - 1) * prop_bin / (prop_bin * accuracies.size()[0] - 1)
            ce += torch.sum(torch.nan_to_num(debias_bins))
        return torch.sqrt(ce) if ce > 0 else torch.tensor(0)
    return ce