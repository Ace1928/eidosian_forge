import logging
import os
from datetime import timedelta
from typing import Dict, List, Optional, Sequence, Union
import pytorch_lightning as pl
from lightning_fabric.utilities.registry import _load_external_callbacks
from pytorch_lightning.callbacks import (
from pytorch_lightning.callbacks.batch_size_finder import BatchSizeFinder
from pytorch_lightning.callbacks.lr_finder import LearningRateFinder
from pytorch_lightning.callbacks.rich_model_summary import RichModelSummary
from pytorch_lightning.callbacks.timer import Timer
from pytorch_lightning.trainer import call
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.rank_zero import rank_zero_info
def _configure_progress_bar(self, enable_progress_bar: bool=True) -> None:
    progress_bars = [c for c in self.trainer.callbacks if isinstance(c, ProgressBar)]
    if len(progress_bars) > 1:
        raise MisconfigurationException('You added multiple progress bar callbacks to the Trainer, but currently only one progress bar is supported.')
    if len(progress_bars) == 1:
        if enable_progress_bar:
            return
        progress_bar_callback = progress_bars[0]
        raise MisconfigurationException(f'Trainer was configured with `enable_progress_bar=False` but found `{progress_bar_callback.__class__.__name__}` in callbacks list.')
    if enable_progress_bar:
        progress_bar_callback = TQDMProgressBar()
        self.trainer.callbacks.append(progress_bar_callback)