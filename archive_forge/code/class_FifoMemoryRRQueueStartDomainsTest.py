import os
from queuelib.rrqueue import RoundRobinQueue
from queuelib.queue import (
from queuelib.tests import QueuelibTestCase, track_closed
class FifoMemoryRRQueueStartDomainsTest(RRQueueStartDomainsTestMixin, QueuelibTestCase):

    def qfactory(self, key):
        return track_closed(FifoMemoryQueue)()