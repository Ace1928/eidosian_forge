import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def _zrange(self, command, dest: Union[KeyT, None], name: KeyT, start: int, end: int, desc: bool=False, byscore: bool=False, bylex: bool=False, withscores: bool=False, score_cast_func: Union[type, Callable, None]=float, offset: Union[int, None]=None, num: Union[int, None]=None) -> ResponseT:
    if byscore and bylex:
        raise DataError('``byscore`` and ``bylex`` can not be specified together.')
    if offset is not None and num is None or (num is not None and offset is None):
        raise DataError('``offset`` and ``num`` must both be specified.')
    if bylex and withscores:
        raise DataError('``withscores`` not supported in combination with ``bylex``.')
    pieces = [command]
    if dest:
        pieces.append(dest)
    pieces.extend([name, start, end])
    if byscore:
        pieces.append('BYSCORE')
    if bylex:
        pieces.append('BYLEX')
    if desc:
        pieces.append('REV')
    if offset is not None and num is not None:
        pieces.extend(['LIMIT', offset, num])
    if withscores:
        pieces.append('WITHSCORES')
    options = {'withscores': withscores, 'score_cast_func': score_cast_func}
    return self.execute_command(*pieces, **options)