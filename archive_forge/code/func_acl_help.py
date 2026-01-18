import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def acl_help(self, **kwargs) -> ResponseT:
    """The ACL HELP command returns helpful text describing
        the different subcommands.

        For more information see https://redis.io/commands/acl-help
        """
    return self.execute_command('ACL HELP', **kwargs)