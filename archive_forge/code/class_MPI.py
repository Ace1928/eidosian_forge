import os
import atexit
import functools
import pickle
import sys
import time
import warnings
import numpy as np
class MPI:
    """Wrapper for MPI world object.

    Decides at runtime (after all imports) which one to use:

    * MPI4Py
    * GPAW
    * a dummy implementation for serial runs

    """

    def __init__(self):
        self.comm = None

    def __getattr__(self, name):
        if self.comm is None:
            self.comm = _get_comm()
        return getattr(self.comm, name)