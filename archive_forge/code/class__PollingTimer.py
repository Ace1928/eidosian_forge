from zope.interface import implementer
from twisted.internet.interfaces import IConsumer, IPushProducer
import pywintypes
import win32api
import win32file
import win32pipe
class _PollingTimer:

    def __init__(self, reactor):
        self.reactor = reactor
        self._resources = []
        self._pollTimer = None
        self._currentTimeout = MAX_TIMEOUT
        self._paused = False

    def _addPollableResource(self, res):
        self._resources.append(res)
        self._checkPollingState()

    def _checkPollingState(self):
        for resource in self._resources:
            if resource.active:
                self._startPolling()
                break
        else:
            self._stopPolling()

    def _startPolling(self):
        if self._pollTimer is None:
            self._pollTimer = self._reschedule()

    def _stopPolling(self):
        if self._pollTimer is not None:
            self._pollTimer.cancel()
            self._pollTimer = None

    def _pause(self):
        self._paused = True

    def _unpause(self):
        self._paused = False
        self._checkPollingState()

    def _reschedule(self):
        if not self._paused:
            return self.reactor.callLater(self._currentTimeout, self._pollEvent)

    def _pollEvent(self):
        workUnits = 0.0
        anyActive = []
        for resource in self._resources:
            if resource.active:
                workUnits += resource.checkWork()
                if resource.active:
                    anyActive.append(resource)
        newTimeout = self._currentTimeout
        if workUnits:
            newTimeout = self._currentTimeout / (workUnits + 1.0)
            if newTimeout < MIN_TIMEOUT:
                newTimeout = MIN_TIMEOUT
        else:
            newTimeout = self._currentTimeout * 2.0
            if newTimeout > MAX_TIMEOUT:
                newTimeout = MAX_TIMEOUT
        self._currentTimeout = newTimeout
        if anyActive:
            self._pollTimer = self._reschedule()