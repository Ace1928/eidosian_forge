from random import random as _goodEnoughRandom
from typing import List
from automat import MethodicalMachine
from twisted.application import service
from twisted.internet import task
from twisted.internet.defer import (
from twisted.logger import Logger
from twisted.python import log
from twisted.python.failure import Failure
class CooperatorService(service.Service):
    """
    Simple L{service.IService} which starts and stops a L{twisted.internet.task.Cooperator}.
    """

    def __init__(self):
        self.coop = task.Cooperator(started=False)

    def coiterate(self, iterator):
        return self.coop.coiterate(iterator)

    def startService(self):
        self.coop.start()

    def stopService(self):
        self.coop.stop()