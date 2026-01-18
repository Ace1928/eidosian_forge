import os
import warnings
from functools import partial, wraps
from typing import Any, Callable
from torchmetrics import _logger as log
def _deprecated_root_import_func(name: str, domain: str) -> None:
    """Warn user that he is importing function from location it has been deprecated."""
    _future_warning(f'Importing `{name}` from `torchmetrics.functional` was deprecated and will be removed in 2.0. Import `{name}` from `torchmetrics.{domain}` instead.')