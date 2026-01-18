from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.regression.utils import _check_data_shape_to_num_outputs
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.data import _bincount, _cumsum, dim_zero_cat
from torchmetrics.utilities.enums import EnumStr
def _calculate_p_value(con_min_dis_pairs: Tensor, n_total: Tensor, preds_ties: Optional[Tensor], preds_ties_p1: Optional[Tensor], preds_ties_p2: Optional[Tensor], target_ties: Optional[Tensor], target_ties_p1: Optional[Tensor], target_ties_p2: Optional[Tensor], variant: _MetricVariant, alternative: Optional[_TestAlternative]) -> Tensor:
    """Calculate p-value for Kendall's tau from metric metadata."""
    t_value_denominator_base = n_total * (n_total - 1) * (2 * n_total + 5)
    if variant == _MetricVariant.A:
        t_value = 3 * con_min_dis_pairs / torch.sqrt(t_value_denominator_base / 2)
    else:
        m = n_total * (n_total - 1)
        t_value_denominator: Tensor = (t_value_denominator_base - preds_ties_p2 - target_ties_p2) / 18
        t_value_denominator += 2 * preds_ties * target_ties / m
        t_value_denominator += preds_ties_p1 * target_ties_p1 / (9 * m * (n_total - 2))
        t_value = con_min_dis_pairs / torch.sqrt(t_value_denominator)
    if alternative == _TestAlternative.TWO_SIDED:
        t_value = torch.abs(t_value)
    if alternative in [_TestAlternative.TWO_SIDED, _TestAlternative.GREATER]:
        t_value *= -1
    p_value = _get_p_value_for_t_value_from_dist(t_value)
    if alternative == _TestAlternative.TWO_SIDED:
        p_value *= 2
    return p_value