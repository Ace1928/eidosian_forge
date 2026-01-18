from oslo_cache.tests.functional import test_base
class TestDogpileCachePyMemcacheBackend(test_base.BaseTestCaseCacheBackend):

    def setUp(self):
        self.config_fixture.config(group='cache', backend='dogpile.cache.pymemcache', memcache_servers='localhost:11212')
        super().setUp()