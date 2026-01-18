import traceback
from ..Qt import QtCore
class RecursiveMutex(Mutex):
    """Mimics threading.RLock class.
    """

    def __init__(self, **kwds):
        kwds['recursive'] = True
        Mutex.__init__(self, **kwds)