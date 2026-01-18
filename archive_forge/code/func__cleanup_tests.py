import os
import itertools
import sys
import weakref
import atexit
import threading        # we want threading to install it's
from subprocess import _args_from_interpreter_flags
from . import process
def _cleanup_tests():
    """Cleanup multiprocessing resources when multiprocessing tests
    completed."""
    from test import support
    process._cleanup()
    from multiprocess import forkserver
    forkserver._forkserver._stop()
    from multiprocess import resource_tracker
    resource_tracker._resource_tracker._stop()
    _run_finalizers()
    support.gc_collect()
    support.reap_children()