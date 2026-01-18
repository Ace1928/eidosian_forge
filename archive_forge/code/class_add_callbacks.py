from __future__ import annotations
from collections.abc import Callable
from contextlib import contextmanager
from typing import ClassVar
class add_callbacks:
    """Context manager for callbacks.

    Takes several callbacks and applies them only in the enclosed context.
    Callbacks can either be represented as a ``Callback`` object, or as a tuple
    of length 4.

    Examples
    --------
    >>> def pretask(key, dsk, state):
    ...     print("Now running {0}").format(key)
    >>> callbacks = (None, pretask, None, None)
    >>> with add_callbacks(callbacks):    # doctest: +SKIP
    ...     res.compute()
    """

    def __init__(self, *callbacks):
        self.callbacks = [normalize_callback(c) for c in callbacks]
        Callback.active.update(self.callbacks)

    def __enter__(self):
        return

    def __exit__(self, type, value, traceback):
        for c in self.callbacks:
            Callback.active.discard(c)