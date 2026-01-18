import os
import sys
import signal
import itertools
import logging
import threading
from _weakrefset import WeakSet
from multiprocessing import process as _mproc
@property
def _authkey(self):
    return self.authkey