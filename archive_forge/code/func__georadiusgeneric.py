import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def _georadiusgeneric(self, command: str, *args: EncodableT, **kwargs: Union[EncodableT, None]) -> ResponseT:
    pieces = list(args)
    if kwargs['unit'] and kwargs['unit'] not in ('m', 'km', 'mi', 'ft'):
        raise DataError('GEORADIUS invalid unit')
    elif kwargs['unit']:
        pieces.append(kwargs['unit'])
    else:
        pieces.append('m')
    if kwargs['any'] and kwargs['count'] is None:
        raise DataError("``any`` can't be provided without ``count``")
    for arg_name, byte_repr in (('withdist', 'WITHDIST'), ('withcoord', 'WITHCOORD'), ('withhash', 'WITHHASH')):
        if kwargs[arg_name]:
            pieces.append(byte_repr)
    if kwargs['count'] is not None:
        pieces.extend(['COUNT', kwargs['count']])
        if kwargs['any']:
            pieces.append('ANY')
    if kwargs['sort']:
        if kwargs['sort'] == 'ASC':
            pieces.append('ASC')
        elif kwargs['sort'] == 'DESC':
            pieces.append('DESC')
        else:
            raise DataError('GEORADIUS invalid sort')
    if kwargs['store'] and kwargs['store_dist']:
        raise DataError('GEORADIUS store and store_dist cant be set together')
    if kwargs['store']:
        pieces.extend([b'STORE', kwargs['store']])
    if kwargs['store_dist']:
        pieces.extend([b'STOREDIST', kwargs['store_dist']])
    return self.execute_command(command, *pieces, **kwargs)