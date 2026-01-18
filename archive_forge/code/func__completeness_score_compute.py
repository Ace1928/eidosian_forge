from typing import Tuple
import torch
from torch import Tensor
from torchmetrics.functional.clustering.mutual_info_score import mutual_info_score
from torchmetrics.functional.clustering.utils import calculate_entropy, check_cluster_labels
def _completeness_score_compute(preds: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
    """Computes the completeness score of a clustering given the predicted and target cluster labels."""
    homogeneity, mutual_info, entropy_preds, _ = _homogeneity_score_compute(preds, target)
    completeness = mutual_info / entropy_preds if entropy_preds else torch.ones_like(entropy_preds)
    return (completeness, homogeneity)