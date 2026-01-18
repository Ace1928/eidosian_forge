import inspect
import logging
from copy import deepcopy
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import torch.nn.utils.prune as pytorch_prune
from lightning_utilities.core.apply_func import apply_to_collection
from torch import Tensor, nn
from typing_extensions import TypedDict, override
import pytorch_lightning as pl
from pytorch_lightning.callbacks.callback import Callback
from pytorch_lightning.core.module import LightningModule
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.rank_zero import rank_zero_debug, rank_zero_only
def _run_pruning(self, current_epoch: int) -> None:
    prune = self._apply_pruning(current_epoch) if callable(self._apply_pruning) else self._apply_pruning
    amount = self.amount(current_epoch) if callable(self.amount) else self.amount
    if not prune or not amount:
        return
    self.apply_pruning(amount)
    if self._use_lottery_ticket_hypothesis(current_epoch) if callable(self._use_lottery_ticket_hypothesis) else self._use_lottery_ticket_hypothesis:
        self.apply_lottery_ticket_hypothesis()