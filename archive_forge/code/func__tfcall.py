import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def _tfcall(self, lib_name: str, func_name: str, keys: KeysT=None, _async: bool=False, *args: List) -> ResponseT:
    pieces = [f'{lib_name}.{func_name}']
    if keys is not None:
        pieces.append(len(keys))
        pieces.extend(keys)
    else:
        pieces.append(0)
    if args is not None:
        pieces.extend(args)
    if _async:
        return self.execute_command('TFCALLASYNC', *pieces)
    return self.execute_command('TFCALL', *pieces)