import math
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, Generator, Optional, Union, cast
from lightning_utilities.core.imports import RequirementCache
from typing_extensions import override
import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress.progress_bar import ProgressBar
from pytorch_lightning.utilities.types import STEP_OUTPUT
class CustomTimeColumn(ProgressColumn):
    max_refresh = 0.5

    def __init__(self, style: Union[str, Style]) -> None:
        self.style = style
        super().__init__()

    def render(self, task: 'Task') -> Text:
        elapsed = task.finished_time if task.finished else task.elapsed
        remaining = task.time_remaining
        elapsed_delta = '-:--:--' if elapsed is None else str(timedelta(seconds=int(elapsed)))
        remaining_delta = '-:--:--' if remaining is None else str(timedelta(seconds=int(remaining)))
        return Text(f'{elapsed_delta} • {remaining_delta}', style=self.style)