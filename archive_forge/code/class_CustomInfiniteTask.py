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
class CustomInfiniteTask(Task):
    """Overrides ``Task`` to define an infinite task.

        This is useful for datasets that do not define a size (infinite size) such as ``IterableDataset``.

        """

    @property
    def time_remaining(self) -> Optional[float]:
        return None