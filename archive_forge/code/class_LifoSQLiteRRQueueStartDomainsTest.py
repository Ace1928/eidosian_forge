import os
from queuelib.rrqueue import RoundRobinQueue
from queuelib.queue import (
from queuelib.tests import QueuelibTestCase, track_closed
class LifoSQLiteRRQueueStartDomainsTest(RRQueueStartDomainsTestMixin, QueuelibTestCase):

    def qfactory(self, key):
        path = os.path.join(self.qdir, str(key))
        return track_closed(LifoSQLiteQueue)(path)