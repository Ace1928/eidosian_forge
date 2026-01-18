import math
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, Generator, Optional, Union, cast
from lightning_utilities.core.imports import RequirementCache
from typing_extensions import override
import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress.progress_bar import ProgressBar
from pytorch_lightning.utilities.types import STEP_OUTPUT
def _reset_progress_bar_ids(self) -> None:
    self.train_progress_bar_id = None
    self.val_sanity_progress_bar_id = None
    self.val_progress_bar_id = None
    self.test_progress_bar_id = None
    self.predict_progress_bar_id = None