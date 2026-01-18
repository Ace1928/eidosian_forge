from __future__ import annotations
from functools import wraps
import inspect
from . import config
from ..util.concurrency import _AsyncUtil
def _maybe_async_provisioning(fn, *args, **kwargs):
    """Run a function in an asyncio loop if any current drivers might need it.

    This function is used for provisioning features that take
    place outside of a specific database driver being selected, so if the
    current driver that happens to be used for the provisioning operation
    is an async driver, it will run in asyncio and not fail.

    Note that for blocking IO database drivers, this means they block the
    event loop.

    """
    if not ENABLE_ASYNCIO:
        return fn(*args, **kwargs)
    if config.any_async:
        return _async_util.run_in_greenlet(fn, *args, **kwargs)
    else:
        return fn(*args, **kwargs)