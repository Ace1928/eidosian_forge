from zope.interface import implementer
from twisted.internet.interfaces import IConsumer, IPushProducer
import pywintypes
import win32api
import win32file
import win32pipe
def _stopPolling(self):
    if self._pollTimer is not None:
        self._pollTimer.cancel()
        self._pollTimer = None