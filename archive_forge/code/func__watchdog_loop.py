import abc
import logging
import threading
import time
from contextlib import contextmanager
from inspect import getframeinfo, stack
from typing import Any, Dict, List, Optional, Set
def _watchdog_loop(self):
    while not self._stop_signaled:
        try:
            self._run_watchdog()
        except Exception:
            log.exception('Error running watchdog')