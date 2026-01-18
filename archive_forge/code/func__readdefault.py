import threading
import sys
from paste.util import filemixin
def _readdefault(self, name, size):
    self._default.read(size)