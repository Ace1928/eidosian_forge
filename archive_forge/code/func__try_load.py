import logging
import os
import random
import time
import urllib
from typing import Any, Callable, Optional, Sized, Tuple, Union
from urllib.error import HTTPError
from warnings import warn
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split
from lightning_fabric.utilities.imports import _IS_WINDOWS
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.imports import _TORCHVISION_AVAILABLE
@staticmethod
def _try_load(path_data: str, trials: int=30, delta: float=1.0) -> Tuple[Tensor, Tensor]:
    """Resolving loading from the same time from multiple concurrent processes."""
    res, exception = (None, None)
    assert trials, 'at least some trial has to be set'
    assert os.path.isfile(path_data), f'missing file: {path_data}'
    for _ in range(trials):
        try:
            res = torch.load(path_data)
        except Exception as ex:
            exception = ex
            time.sleep(delta * random.random())
        else:
            break
    assert res is not None
    if exception is not None:
        raise exception
    return res