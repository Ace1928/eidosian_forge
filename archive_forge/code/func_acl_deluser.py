import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def acl_deluser(self, *username: str, **kwargs) -> ResponseT:
    """
        Delete the ACL for the specified ``username``s

        For more information see https://redis.io/commands/acl-deluser
        """
    return self.execute_command('ACL DELUSER', *username, **kwargs)