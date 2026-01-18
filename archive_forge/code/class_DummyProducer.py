from twisted.protocols import pcp
from twisted.trial import unittest
class DummyProducer:
    resumed = False
    stopped = False
    paused = False

    def __init__(self, consumer):
        self.consumer = consumer

    def resumeProducing(self):
        self.resumed = True
        self.paused = False

    def pauseProducing(self):
        self.paused = True

    def stopProducing(self):
        self.stopped = True