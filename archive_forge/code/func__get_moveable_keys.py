from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union
from redis.exceptions import RedisError, ResponseError
from redis.utils import str_if_bytes
def _get_moveable_keys(self, redis_conn, *args):
    """
        NOTE: Due to a bug in redis<7.0, this function does not work properly
        for EVAL or EVALSHA when the `numkeys` arg is 0.
         - issue: https://github.com/redis/redis/issues/9493
         - fix: https://github.com/redis/redis/pull/9733

        So, don't use this function with EVAL or EVALSHA.
        """
    pieces = args[0].split() + list(args[1:])
    try:
        keys = redis_conn.execute_command('COMMAND GETKEYS', *pieces)
    except ResponseError as e:
        message = e.__str__()
        if 'Invalid arguments' in message or 'The command has no key arguments' in message:
            return None
        else:
            raise e
    return keys