import threading
import sys
from paste.util import filemixin
def _readfactory(self, name, size):
    self._factory(name).read(size)