from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.six import PY2
class FakeURLLIB3(object):

    def __init__(self):
        self._collections = self
        self.poolmanager = self
        self.connection = self
        self.connectionpool = self
        self.RecentlyUsedContainer = object()
        self.PoolManager = object()
        self.match_hostname = object()
        self.HTTPConnectionPool = _HTTPConnectionPool