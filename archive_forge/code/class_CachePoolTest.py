import datetime
import os
import time
from unittest import mock
import uuid
import fixtures
from keystoneauth1 import exceptions as ksa_exceptions
from keystoneauth1 import fixture
from keystoneauth1 import loading
from keystoneauth1 import session
import oslo_cache
from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import timeutils
import pbr.version
import testresources
from testtools import matchers
import webob
import webob.dec
from keystonemiddleware import auth_token
from keystonemiddleware.auth_token import _base
from keystonemiddleware.auth_token import _cache
from keystonemiddleware.auth_token import _exceptions as ksm_exceptions
from keystonemiddleware.tests.unit.auth_token import base
from keystonemiddleware.tests.unit import client_fixtures
class CachePoolTest(BaseAuthTokenMiddlewareTest):

    def test_use_cache_from_env(self):
        env = {'swift.cache': 'CACHE_TEST'}
        conf = {'cache': 'swift.cache'}
        self.set_middleware(conf=conf)
        self.middleware._token_cache.initialize(env)
        with self.middleware._token_cache._cache_pool.reserve() as cache:
            self.assertEqual(cache, 'CACHE_TEST')

    def test_not_use_cache_from_env(self):
        self.set_middleware()
        env = {'swift.cache': 'CACHE_TEST'}
        self.middleware._token_cache.initialize(env)
        with self.middleware._token_cache._cache_pool.reserve() as cache:
            self.assertNotEqual(cache, 'CACHE_TEST')

    def test_multiple_context_managers_share_single_client(self):
        self.set_middleware()
        token_cache = self.middleware._token_cache
        env = {}
        token_cache.initialize(env)
        caches = []
        with token_cache._cache_pool.reserve() as cache:
            caches.append(cache)
        with token_cache._cache_pool.reserve() as cache:
            caches.append(cache)
        self.assertIs(caches[0], caches[1])
        self.assertEqual(set(caches), set(token_cache._cache_pool))

    def test_nested_context_managers_create_multiple_clients(self):
        self.set_middleware()
        env = {}
        self.middleware._token_cache.initialize(env)
        token_cache = self.middleware._token_cache
        with token_cache._cache_pool.reserve() as outer_cache:
            with token_cache._cache_pool.reserve() as inner_cache:
                self.assertNotEqual(outer_cache, inner_cache)
        self.assertEqual(set([inner_cache, outer_cache]), set(token_cache._cache_pool))