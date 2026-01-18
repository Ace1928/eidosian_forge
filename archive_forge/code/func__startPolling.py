from zope.interface import implementer
from twisted.internet.interfaces import IConsumer, IPushProducer
import pywintypes
import win32api
import win32file
import win32pipe
def _startPolling(self):
    if self._pollTimer is None:
        self._pollTimer = self._reschedule()