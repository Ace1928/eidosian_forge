import functools
import threading
from contextlib import contextmanager
from .driver import driver, USE_NV_BINDING
def _destroy_all_contexts(self):
    for gpu in self.gpus:
        gpu.reset()