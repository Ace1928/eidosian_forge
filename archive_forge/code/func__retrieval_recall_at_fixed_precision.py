from typing import Any, Callable, List, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics import Metric
from torchmetrics.functional.retrieval.precision_recall_curve import retrieval_precision_recall_curve
from torchmetrics.retrieval.base import _retrieval_aggregate
from torchmetrics.utilities.checks import _check_retrieval_inputs
from torchmetrics.utilities.data import _flexible_bincount, dim_zero_cat
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE, plot_curve
def _retrieval_recall_at_fixed_precision(precision: Tensor, recall: Tensor, top_k: Tensor, min_precision: float) -> Tuple[Tensor, Tensor]:
    """Compute maximum recall with condition that corresponding precision >= `min_precision`.

    Args:
        top_k: tensor with all possible k
        precision: tensor with all values precisions@k for k from top_k tensor
        recall: tensor with all values recall@k for k from top_k tensor
        min_precision: float value specifying minimum precision threshold.

    Returns:
        Maximum recall value, corresponding it best k

    """
    try:
        max_recall, best_k = max(((r, k) for p, r, k in zip(precision, recall, top_k) if p >= min_precision))
    except ValueError:
        max_recall = torch.tensor(0.0, device=recall.device, dtype=recall.dtype)
        best_k = torch.tensor(len(top_k))
    if max_recall == 0.0:
        best_k = torch.tensor(len(top_k), device=top_k.device, dtype=top_k.dtype)
    return (max_recall, best_k)