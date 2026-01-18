import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def acl_log(self, count: Union[int, None]=None, **kwargs) -> ResponseT:
    """
        Get ACL logs as a list.
        :param int count: Get logs[0:count].
        :rtype: List.

        For more information see https://redis.io/commands/acl-log
        """
    args = []
    if count is not None:
        if not isinstance(count, int):
            raise DataError('ACL LOG count must be an integer')
        args.append(count)
    return self.execute_command('ACL LOG', *args, **kwargs)