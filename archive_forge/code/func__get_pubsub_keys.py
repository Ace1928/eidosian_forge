from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union
from redis.exceptions import RedisError, ResponseError
from redis.utils import str_if_bytes
def _get_pubsub_keys(self, *args):
    """
        Get the keys from pubsub command.
        Although PubSub commands have predetermined key locations, they are not
        supported in the 'COMMAND's output, so the key positions are hardcoded
        in this method
        """
    if len(args) < 2:
        return None
    args = [str_if_bytes(arg) for arg in args]
    command = args[0].upper()
    keys = None
    if command == 'PUBSUB':
        pubsub_type = args[1].upper()
        if pubsub_type in ['CHANNELS', 'NUMSUB', 'SHARDCHANNELS', 'SHARDNUMSUB']:
            keys = args[2:]
    elif command in ['SUBSCRIBE', 'PSUBSCRIBE', 'UNSUBSCRIBE', 'PUNSUBSCRIBE']:
        keys = list(args[1:])
    elif command in ['PUBLISH', 'SPUBLISH']:
        keys = [args[1]]
    return keys