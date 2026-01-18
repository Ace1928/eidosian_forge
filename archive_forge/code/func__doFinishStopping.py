from random import random as _goodEnoughRandom
from typing import List
from automat import MethodicalMachine
from twisted.application import service
from twisted.internet import task
from twisted.internet.defer import (
from twisted.logger import Logger
from twisted.python import log
from twisted.python.failure import Failure
def _doFinishStopping(self):
    self._stopWaiters, waiting = ([], self._stopWaiters)
    for w in waiting:
        w.callback(None)