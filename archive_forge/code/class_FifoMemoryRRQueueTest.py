import os
from queuelib.rrqueue import RoundRobinQueue
from queuelib.queue import (
from queuelib.tests import QueuelibTestCase, track_closed
class FifoMemoryRRQueueTest(RRQueueTestMixin, FifoTestMixin, QueuelibTestCase):

    def qfactory(self, key):
        return track_closed(FifoMemoryQueue)()