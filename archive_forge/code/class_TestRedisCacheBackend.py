from oslo_cache.tests.functional import test_base
class TestRedisCacheBackend(test_base.BaseTestCaseCacheBackend):

    def setUp(self):
        self.config_fixture.config(group='cache', backend='dogpile.cache.redis', redis_server='127.0.0.1:6379')
        super().setUp()