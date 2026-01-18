from typing import Optional, Tuple
from torch import Tensor
from torchmetrics.functional.retrieval.average_precision import retrieval_average_precision
from torchmetrics.functional.retrieval.fall_out import retrieval_fall_out
from torchmetrics.functional.retrieval.hit_rate import retrieval_hit_rate
from torchmetrics.functional.retrieval.ndcg import retrieval_normalized_dcg
from torchmetrics.functional.retrieval.precision import retrieval_precision
from torchmetrics.functional.retrieval.precision_recall_curve import retrieval_precision_recall_curve
from torchmetrics.functional.retrieval.r_precision import retrieval_r_precision
from torchmetrics.functional.retrieval.recall import retrieval_recall
from torchmetrics.functional.retrieval.reciprocal_rank import retrieval_reciprocal_rank
from torchmetrics.utilities.prints import _deprecated_root_import_func
def _retrieval_r_precision(preds: Tensor, target: Tensor) -> Tensor:
    """Wrapper for deprecated import.

    >>> from torch import tensor
    >>> preds = tensor([0.2, 0.3, 0.5])
    >>> target = tensor([True, False, True])
    >>> _retrieval_r_precision(preds, target)
    tensor(0.5000)

    """
    _deprecated_root_import_func('retrieval_r_precision', 'retrieval')
    return retrieval_r_precision(preds=preds, target=target)