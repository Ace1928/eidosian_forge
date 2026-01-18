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
class TestBMemcacheClient(test_cache.BaseTestCase):

    def test_can_create_with_kwargs(self):
        client = _bmemcache_pool._BMemcacheClient('foo', password='123456')
        self.assertEqual('123456', client.password)
        self.assertIsInstance(client, _bmemcache_pool._BMemcacheClient)