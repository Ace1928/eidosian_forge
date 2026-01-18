import sys
import warnings
from contextlib import contextmanager
from functools import lru_cache as _lru_cache
from typing import Any
from torch.backends import __allow_nonbracketed_mutation, ContextProp, PropModule
def _set_enabled(_enabled: bool) -> None:
    if not is_available() and _enabled:
        raise ValueError(f'opt_einsum is not available, so setting `enabled` to {_enabled} will not reap the benefits of calculating an optimal path for einsum. torch.einsum will fall back to contracting from left to right. To enable this optimal path calculation, please install opt-einsum.')
    global enabled
    enabled = _enabled