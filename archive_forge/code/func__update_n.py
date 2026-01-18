import importlib
import math
import os
import sys
from typing import Any, Dict, Optional, Union
from typing_extensions import override
from pytorch_lightning.utilities.types import STEP_OUTPUT
import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress.progress_bar import ProgressBar
from pytorch_lightning.utilities.rank_zero import rank_zero_debug
def _update_n(bar: _tqdm, value: int) -> None:
    if not bar.disable:
        bar.n = value
        bar.refresh()