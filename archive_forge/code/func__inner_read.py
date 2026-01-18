import io
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional, Union
from tqdm.auto import tqdm as old_tqdm
from ..constants import HF_HUB_DISABLE_PROGRESS_BARS
def _inner_read(size: Optional[int]=-1) -> bytes:
    data = f_read(size)
    pbar.update(len(data))
    return data