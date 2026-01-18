from oslo_cache.tests.functional import test_base
class TestRedisSentinelCacheBackend(test_base.BaseTestCaseCacheBackend):

    def setUp(self):
        self.config_fixture.config(group='cache', backend='dogpile.cache.redis_sentinel', redis_sentinels=['127.0.0.1:6380'], redis_sentinel_service_name='pifpaf')
        super().setUp()