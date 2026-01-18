import atexit
import functools
import os
import pathlib
import sys
from types import TracebackType
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple, Type, Union
from urllib.parse import quote
import sentry_sdk  # type: ignore
import sentry_sdk.utils  # type: ignore
import wandb
import wandb.env
import wandb.util
def _safe_noop(func: Callable) -> Callable:
    """Decorator to ensure that Sentry methods do nothing if disabled and don't raise."""

    @functools.wraps(func)
    def wrapper(self: Type['Sentry'], *args: Any, **kwargs: Any) -> Any:
        if self._disabled:
            return None
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            if func.__name__ != 'exception':
                self.exception(f'Error in {func.__name__}: {e}')
            return None
    return wrapper