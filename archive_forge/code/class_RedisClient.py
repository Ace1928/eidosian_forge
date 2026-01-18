import enum
import functools
import redis
from redis import exceptions as redis_exceptions
class RedisClient(redis.Redis):
    """A redis client that can be closed (and raises on-usage after closed).

    TODO(harlowja): if https://github.com/andymccurdy/redis-py/issues/613 ever
    gets resolved or merged or other then we can likely remove this.
    """

    def __init__(self, *args, **kwargs):
        super(RedisClient, self).__init__(*args, **kwargs)
        self.closed = False

    def close(self):
        self.closed = True
        self.connection_pool.disconnect()
    execute_command = _raise_on_closed(redis.Redis.execute_command)
    transaction = _raise_on_closed(redis.Redis.transaction)
    pubsub = _raise_on_closed(redis.Redis.pubsub)