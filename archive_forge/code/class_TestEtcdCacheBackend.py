from oslo_cache.tests.functional import test_base
class TestEtcdCacheBackend(test_base.BaseTestCaseCacheBackend):

    def setUp(self):
        self.config_fixture.config(group='cache', backend='oslo_cache.etcd3gw', backend_argument=['host:127.0.0.1', 'port:2379'])
        super().setUp()