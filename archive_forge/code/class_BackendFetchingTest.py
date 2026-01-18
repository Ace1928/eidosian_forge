import contextlib
from zake import fake_client
from taskflow.jobs import backends
from taskflow.jobs.backends import impl_redis
from taskflow.jobs.backends import impl_zookeeper
from taskflow import test
class BackendFetchingTest(test.TestCase):

    def test_zk_entry_point_text(self):
        conf = 'zookeeper'
        with contextlib.closing(backends.fetch('test', conf)) as be:
            self.assertIsInstance(be, impl_zookeeper.ZookeeperJobBoard)

    def test_zk_entry_point(self):
        conf = {'board': 'zookeeper'}
        with contextlib.closing(backends.fetch('test', conf)) as be:
            self.assertIsInstance(be, impl_zookeeper.ZookeeperJobBoard)

    def test_zk_entry_point_existing_client(self):
        existing_client = fake_client.FakeClient()
        conf = {'board': 'zookeeper'}
        kwargs = {'client': existing_client}
        with contextlib.closing(backends.fetch('test', conf, **kwargs)) as be:
            self.assertIsInstance(be, impl_zookeeper.ZookeeperJobBoard)
            self.assertIs(existing_client, be._client)

    def test_redis_entry_point_text(self):
        conf = 'redis'
        with contextlib.closing(backends.fetch('test', conf)) as be:
            self.assertIsInstance(be, impl_redis.RedisJobBoard)

    def test_redis_entry_point(self):
        conf = {'board': 'redis'}
        with contextlib.closing(backends.fetch('test', conf)) as be:
            self.assertIsInstance(be, impl_redis.RedisJobBoard)