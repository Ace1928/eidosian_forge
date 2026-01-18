import threading
import sys
from paste.util import filemixin
def _writeerror(self, name, v):
    assert False, 'There is no PrintCatcher output stream for the thread %r' % name