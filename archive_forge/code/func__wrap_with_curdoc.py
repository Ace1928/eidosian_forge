from __future__ import annotations
import logging # isort:skip
import weakref
from collections import defaultdict
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable
from ..core.enums import HoldPolicy, HoldPolicyType
from ..events import (
from ..model import Model
from ..models.callbacks import Callback as JSEventCallback
from ..util.callback_manager import _check_callback
from .events import (
from .locking import UnlockedDocumentProxy
def _wrap_with_curdoc(doc: Document, f: Callable[..., Any]) -> Callable[..., Any]:

    @wraps(f)
    def wrapper(*args: Any, **kwargs: Any) -> None:

        @wraps(f)
        def invoke() -> Any:
            return f(*args, **kwargs)
        return invoke_with_curdoc(doc, invoke)
    return wrapper