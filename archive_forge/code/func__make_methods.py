import threading
import sys
import tempfile
import time
from . import context
from . import process
from . import util
def _make_methods(self):
    self.acquire = self._lock.acquire
    self.release = self._lock.release