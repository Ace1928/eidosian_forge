from typing import Optional, Tuple
import torch
from torch import Tensor
from torchmetrics.functional.classification.confusion_matrix import (
from torchmetrics.utilities.data import _cumsum
def _multilabel_ranking_loss_update(preds: Tensor, target: Tensor) -> Tuple[Tensor, int]:
    """Accumulate state for label ranking loss.

    Args:
        preds: tensor with predictions
        target: tensor with ground truth labels
        sample_weight: optional tensor with weight for each sample

    """
    num_preds, num_labels = preds.shape
    relevant = target == 1
    num_relevant = relevant.sum(dim=1)
    mask = (num_relevant > 0) & (num_relevant < num_labels)
    preds = preds[mask]
    relevant = relevant[mask]
    num_relevant = num_relevant[mask]
    if len(preds) == 0:
        return (torch.tensor(0.0, device=preds.device), 1)
    inverse = preds.argsort(dim=1).argsort(dim=1)
    per_label_loss = ((num_labels - inverse) * relevant).to(torch.float32)
    correction = 0.5 * num_relevant * (num_relevant + 1)
    denom = num_relevant * (num_labels - num_relevant)
    loss = (per_label_loss.sum(dim=1) - correction) / denom
    return (loss.sum(), num_preds)