import logging
from typing import Any, Callable, Dict, Optional, Tuple
import torch
from torch import Tensor
from typing_extensions import override
import pytorch_lightning as pl
from pytorch_lightning.callbacks.callback import Callback
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.rank_zero import rank_prefixed_message, rank_zero_warn
def _should_skip_check(self, trainer: 'pl.Trainer') -> bool:
    from pytorch_lightning.trainer.states import TrainerFn
    return trainer.state.fn != TrainerFn.FITTING or trainer.sanity_checking