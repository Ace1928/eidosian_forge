import collections
import logging
import threading
import time
import types
class BrokenExecutor(RuntimeError):
    """
    Raised when a executor has become non-functional after a severe failure.
    """