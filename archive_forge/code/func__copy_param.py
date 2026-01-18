import inspect
import logging
from copy import deepcopy
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import torch.nn.utils.prune as pytorch_prune
from lightning_utilities.core.apply_func import apply_to_collection
from torch import Tensor, nn
from typing_extensions import TypedDict, override
import pytorch_lightning as pl
from pytorch_lightning.callbacks.callback import Callback
from pytorch_lightning.core.module import LightningModule
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.rank_zero import rank_zero_debug, rank_zero_only
@staticmethod
def _copy_param(new: nn.Module, old: nn.Module, name: str) -> None:
    dst = getattr(new, name)
    src = getattr(old, name)
    if dst is None or src is None or (not isinstance(dst, Tensor)) or (not isinstance(src, Tensor)):
        return
    dst.data = src.data.to(dst.device)