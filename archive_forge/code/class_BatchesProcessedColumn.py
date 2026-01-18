import math
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, Generator, Optional, Union, cast
from lightning_utilities.core.imports import RequirementCache
from typing_extensions import override
import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress.progress_bar import ProgressBar
from pytorch_lightning.utilities.types import STEP_OUTPUT
class BatchesProcessedColumn(ProgressColumn):

    def __init__(self, style: Union[str, Style]):
        self.style = style
        super().__init__()

    def render(self, task: 'Task') -> RenderableType:
        total = task.total if task.total != float('inf') else '--'
        return Text(f'{int(task.completed)}/{total}', style=self.style)