from __future__ import annotations
from typing import Literal
from zope.interface import implementer
from twisted.internet.interfaces import IPushProducer
from twisted.internet.protocol import Protocol
from twisted.internet.task import Clock
from twisted.test.iosim import FakeTransport, connect, connectedServerAndClient
from twisted.trial.unittest import TestCase
@implementer(IPushProducer)
class StrictPushProducer:
    """
    An L{IPushProducer} implementation which produces nothing but enforces
    preconditions on its state transition methods.
    """
    _state = 'running'

    def stopProducing(self) -> None:
        if self._state == 'stopped':
            raise ValueError('Cannot stop already-stopped IPushProducer')
        self._state = 'stopped'

    def pauseProducing(self) -> None:
        if self._state != 'running':
            raise ValueError(f'Cannot pause {self._state} IPushProducer')
        self._state = 'paused'

    def resumeProducing(self) -> None:
        if self._state != 'paused':
            raise ValueError(f'Cannot resume {self._state} IPushProducer')
        self._state = 'running'