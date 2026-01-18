from random import random as _goodEnoughRandom
from typing import List
from automat import MethodicalMachine
from twisted.application import service
from twisted.internet import task
from twisted.internet.defer import (
from twisted.logger import Logger
from twisted.python import log
from twisted.python.failure import Failure
def _doWait(self):
    self._failedAttempts += 1
    delay = self._timeoutForAttempt(self._failedAttempts)
    self._log.info('Scheduling retry {attempt} to connect {endpoint} in {delay} seconds.', attempt=self._failedAttempts, endpoint=self._endpoint, delay=delay)
    self._retryCall = self._clock.callLater(delay, self._reconnect)