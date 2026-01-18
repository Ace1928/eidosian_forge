import atexit
from threading import Event, Thread, current_thread
from time import time
from warnings import warn
class TqdmSynchronisationWarning(RuntimeWarning):
    """tqdm multi-thread/-process errors which may cause incorrect nesting
    but otherwise no adverse effects"""
    pass