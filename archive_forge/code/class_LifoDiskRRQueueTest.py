import os
from queuelib.rrqueue import RoundRobinQueue
from queuelib.queue import (
from queuelib.tests import QueuelibTestCase, track_closed
class LifoDiskRRQueueTest(RRQueueTestMixin, LifoTestMixin, DiskTestMixin, QueuelibTestCase):

    def qfactory(self, key):
        path = os.path.join(self.qdir, str(key))
        return track_closed(LifoDiskQueue)(path)