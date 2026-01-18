import abc
import logging
import threading
import time
from contextlib import contextmanager
from inspect import getframeinfo, stack
from typing import Any, Dict, List, Optional, Set
def _reap_worker_no_throw(self, worker_id: Any) -> bool:
    """
        Wraps ``_reap_worker(worker_id)``, if an uncaught exception is
        thrown, then it considers the worker as reaped.
        """
    try:
        return self._reap_worker(worker_id)
    except Exception:
        log.exception('Uncaught exception thrown from _reap_worker(), check that the implementation correctly catches exceptions')
        return True