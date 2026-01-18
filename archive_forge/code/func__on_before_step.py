from collections import OrderedDict
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any, Dict
from torch import Tensor
from typing_extensions import override
import pytorch_lightning as pl
from pytorch_lightning.core.optimizer import do_nothing_closure
from pytorch_lightning.loops import _Loop
from pytorch_lightning.loops.optimization.closure import OutputResult
from pytorch_lightning.loops.progress import _Progress, _ReadyCompletedTracker
from pytorch_lightning.trainer import call
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.types import STEP_OUTPUT
def _on_before_step(self) -> None:
    self.optim_step_progress.increment_ready()
    self.trainer.profiler.start('optimizer_step')