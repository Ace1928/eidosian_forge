from __future__ import annotations
import multiprocessing
import threading
import uuid
import weakref
from collections.abc import Hashable, MutableMapping
from typing import Any, ClassVar
from weakref import WeakValueDictionary
def _get_scheduler(get=None, collection=None) -> str | None:
    """Determine the dask scheduler that is being used.

    None is returned if no dask scheduler is active.

    See Also
    --------
    dask.base.get_scheduler
    """
    try:
        import dask
        from dask.base import get_scheduler
        actual_get = get_scheduler(get, collection)
    except ImportError:
        return None
    try:
        from dask.distributed import Client
        if isinstance(actual_get.__self__, Client):
            return 'distributed'
    except (ImportError, AttributeError):
        pass
    try:
        if actual_get is dask.multiprocessing.get:
            return 'multiprocessing'
    except AttributeError:
        pass
    return 'threaded'