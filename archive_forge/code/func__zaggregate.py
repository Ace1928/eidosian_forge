import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def _zaggregate(self, command: str, dest: Union[KeyT, None], keys: Union[Sequence[KeyT], Mapping[AnyKeyT, float]], aggregate: Union[str, None]=None, **options) -> ResponseT:
    pieces: list[EncodableT] = [command]
    if dest is not None:
        pieces.append(dest)
    pieces.append(len(keys))
    if isinstance(keys, dict):
        keys, weights = (keys.keys(), keys.values())
    else:
        weights = None
    pieces.extend(keys)
    if weights:
        pieces.append(b'WEIGHTS')
        pieces.extend(weights)
    if aggregate:
        if aggregate.upper() in ['SUM', 'MIN', 'MAX']:
            pieces.append(b'AGGREGATE')
            pieces.append(aggregate)
        else:
            raise DataError('aggregate can be sum, min or max.')
    if options.get('withscores', False):
        pieces.append(b'WITHSCORES')
    return self.execute_command(*pieces, **options)