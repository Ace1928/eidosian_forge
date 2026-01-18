from zope.interface import implementer
from twisted.internet.interfaces import IConsumer, IPushProducer
import pywintypes
import win32api
import win32file
import win32pipe
def _checkPollingState(self):
    for resource in self._resources:
        if resource.active:
            self._startPolling()
            break
    else:
        self._stopPolling()