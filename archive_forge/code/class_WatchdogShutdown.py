from __future__ import annotations
import sys
import threading
from typing import TYPE_CHECKING
class WatchdogShutdown(Exception):
    """
    Semantic exception used to signal an external shutdown event.
    """
    pass