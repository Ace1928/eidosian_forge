from typing import Dict, List, Optional, Tuple
import torch
from typing_extensions import Literal
from torchmetrics.functional.classification.stat_scores import (
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.compute import _safe_divide
from torchmetrics.utilities.data import _flexible_bincount
def _compute_binary_demographic_parity(tp: torch.Tensor, fp: torch.Tensor, tn: torch.Tensor, fn: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Compute demographic parity based on the binary stats."""
    pos_rates = _safe_divide(tp + fp, tp + fp + tn + fn)
    min_pos_rate_id = torch.argmin(pos_rates)
    max_pos_rate_id = torch.argmax(pos_rates)
    return {f'DP_{min_pos_rate_id}_{max_pos_rate_id}': _safe_divide(pos_rates[min_pos_rate_id], pos_rates[max_pos_rate_id])}