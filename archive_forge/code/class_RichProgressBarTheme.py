import math
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, Generator, Optional, Union, cast
from lightning_utilities.core.imports import RequirementCache
from typing_extensions import override
import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress.progress_bar import ProgressBar
from pytorch_lightning.utilities.types import STEP_OUTPUT
@dataclass
class RichProgressBarTheme:
    """Styles to associate to different base components.

    Args:
        description: Style for the progress bar description. For eg., Epoch x, Testing, etc.
        progress_bar: Style for the bar in progress.
        progress_bar_finished: Style for the finished progress bar.
        progress_bar_pulse: Style for the progress bar when `IterableDataset` is being processed.
        batch_progress: Style for the progress tracker (i.e 10/50 batches completed).
        time: Style for the processed time and estimate time remaining.
        processing_speed: Style for the speed of the batches being processed.
        metrics: Style for the metrics

    https://rich.readthedocs.io/en/stable/style.html

    """
    description: Union[str, Style] = 'white'
    progress_bar: Union[str, Style] = '#6206E0'
    progress_bar_finished: Union[str, Style] = '#6206E0'
    progress_bar_pulse: Union[str, Style] = '#6206E0'
    batch_progress: Union[str, Style] = 'white'
    time: Union[str, Style] = 'grey54'
    processing_speed: Union[str, Style] = 'grey70'
    metrics: Union[str, Style] = 'white'
    metrics_text_delimiter: str = ' '
    metrics_format: str = '.3f'