import os
from queuelib.pqueue import PriorityQueue
from queuelib.queue import (
from queuelib.tests import QueuelibTestCase, track_closed
class FifoMemoryPriorityQueueTest(PQueueTestMixin, FifoTestMixin, QueuelibTestCase):

    def qfactory(self, prio):
        return track_closed(FifoMemoryQueue)()