from typing import Tuple
import torch
from torch import Tensor
from torchmetrics.functional.clustering.mutual_info_score import mutual_info_score
from torchmetrics.functional.clustering.utils import calculate_entropy, check_cluster_labels
def _homogeneity_score_compute(preds: Tensor, target: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Computes the homogeneity score of a clustering given the predicted and target cluster labels."""
    check_cluster_labels(preds, target)
    if len(target) == 0:
        zero = torch.tensor(0.0, dtype=torch.float32, device=preds.device)
        return (zero.clone(), zero.clone(), zero.clone(), zero.clone())
    entropy_target = calculate_entropy(target)
    entropy_preds = calculate_entropy(preds)
    mutual_info = mutual_info_score(preds, target)
    homogeneity = mutual_info / entropy_target if entropy_target else torch.ones_like(entropy_target)
    return (homogeneity, mutual_info, entropy_preds, entropy_target)