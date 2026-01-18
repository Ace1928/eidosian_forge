import functools
import inspect
import os
from collections import OrderedDict
from contextlib import contextmanager
from functools import partial
from typing import Any, Callable, Dict, Generator, Iterable, Optional, Sized, Tuple, Type, Union
from lightning_utilities.core.inheritance import get_all_subclasses
from torch.utils.data import BatchSampler, DataLoader, IterableDataset, Sampler
from typing_extensions import TypeGuard
from lightning_fabric.utilities.enums import LightningEnum
from lightning_fabric.utilities.exceptions import MisconfigurationException
from lightning_fabric.utilities.rank_zero import rank_zero_warn
from lightning_fabric.utilities.seed import pl_worker_init_function
class _WrapAttrTag(LightningEnum):
    SET = 'set'
    DEL = 'del'

    def __call__(self, *args: Any) -> None:
        fn: Union[Callable[[object, str], None], Callable[[object, str, Any], None]]
        fn = setattr if self == self.SET else delattr
        return fn(*args)