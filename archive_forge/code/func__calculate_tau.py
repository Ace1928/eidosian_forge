from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.regression.utils import _check_data_shape_to_num_outputs
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.data import _bincount, _cumsum, dim_zero_cat
from torchmetrics.utilities.enums import EnumStr
def _calculate_tau(preds: Tensor, target: Tensor, concordant_pairs: Tensor, discordant_pairs: Tensor, con_min_dis_pairs: Tensor, n_total: Tensor, preds_ties: Optional[Tensor], target_ties: Optional[Tensor], variant: _MetricVariant) -> Tensor:
    """Calculate Kendall's tau from metric metadata."""
    if variant == _MetricVariant.A:
        return con_min_dis_pairs / (concordant_pairs + discordant_pairs)
    if variant == _MetricVariant.B:
        total_combinations: Tensor = n_total * (n_total - 1) // 2
        denominator = (total_combinations - preds_ties) * (total_combinations - target_ties)
        return con_min_dis_pairs / torch.sqrt(denominator)
    preds_unique = torch.tensor([len(p.unique()) for p in preds.T], dtype=preds.dtype, device=preds.device)
    target_unique = torch.tensor([len(t.unique()) for t in target.T], dtype=target.dtype, device=target.device)
    min_classes = torch.minimum(preds_unique, target_unique)
    return 2 * con_min_dis_pairs / ((min_classes - 1) / min_classes * n_total ** 2)