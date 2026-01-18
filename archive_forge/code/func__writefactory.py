import threading
import sys
from paste.util import filemixin
def _writefactory(self, name, v):
    self._factory(name).write(v)