import logging
from typing import Any, Callable, Dict, Optional, Tuple
import torch
from torch import Tensor
from typing_extensions import override
import pytorch_lightning as pl
from pytorch_lightning.callbacks.callback import Callback
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.rank_zero import rank_prefixed_message, rank_zero_warn
def _improvement_message(self, current: Tensor) -> str:
    """Formats a log message that informs the user about an improvement in the monitored score."""
    if torch.isfinite(self.best_score):
        msg = f'Metric {self.monitor} improved by {abs(self.best_score - current):.3f} >= min_delta = {abs(self.min_delta)}. New best score: {current:.3f}'
    else:
        msg = f'Metric {self.monitor} improved. New best score: {current:.3f}'
    return msg