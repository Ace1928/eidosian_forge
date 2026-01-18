import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def _evalsha(self, command: str, sha: str, numkeys: int, *keys_and_args: list) -> Union[Awaitable[str], str]:
    return self.execute_command(command, sha, numkeys, *keys_and_args)