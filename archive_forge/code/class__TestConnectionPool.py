import queue
import threading
import time
from unittest import mock
import testtools
from testtools import matchers
from oslo_cache import _bmemcache_pool
from oslo_cache import _memcache_pool
from oslo_cache import exception
from oslo_cache.tests import test_cache
class _TestConnectionPool(_memcache_pool.ConnectionPool):
    destroyed_value = 'destroyed'

    def _create_connection(self):
        return mock.MagicMock()

    def _destroy_connection(self, conn):
        conn(self.destroyed_value)