import threading
import sys
from paste.util import filemixin
def _readerror(self, name, size):
    assert False, 'There is no StdinCatcher output stream for the thread %r' % name