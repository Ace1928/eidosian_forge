import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def _fcall(self, command: str, function, numkeys: int, *keys_and_args: Optional[List]) -> Union[Awaitable[str], str]:
    return self.execute_command(command, function, numkeys, *keys_and_args)