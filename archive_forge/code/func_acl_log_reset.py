import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def acl_log_reset(self, **kwargs) -> ResponseT:
    """
        Reset ACL logs.
        :rtype: Boolean.

        For more information see https://redis.io/commands/acl-log
        """
    args = [b'RESET']
    return self.execute_command('ACL LOG', *args, **kwargs)