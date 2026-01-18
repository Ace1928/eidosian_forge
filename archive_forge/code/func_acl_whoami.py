import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def acl_whoami(self, **kwargs) -> ResponseT:
    """Get the username for the current connection

        For more information see https://redis.io/commands/acl-whoami
        """
    return self.execute_command('ACL WHOAMI', **kwargs)