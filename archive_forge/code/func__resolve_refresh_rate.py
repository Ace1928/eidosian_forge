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
@staticmethod
def _resolve_refresh_rate(refresh_rate: int) -> int:
    if os.getenv('COLAB_GPU') and refresh_rate == 1:
        rank_zero_debug('Using a higher refresh rate on Colab. Setting it to `20`')
        return 20
    if 'TQDM_MINITERS' in os.environ:
        return max(int(os.environ['TQDM_MINITERS']), refresh_rate)
    return refresh_rate