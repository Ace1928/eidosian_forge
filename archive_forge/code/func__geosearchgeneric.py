import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def _geosearchgeneric(self, command: str, *args: EncodableT, **kwargs: Union[EncodableT, None]) -> ResponseT:
    pieces = list(args)
    if kwargs['member'] is None:
        if kwargs['longitude'] is None or kwargs['latitude'] is None:
            raise DataError('GEOSEARCH must have member or longitude and latitude')
    if kwargs['member']:
        if kwargs['longitude'] or kwargs['latitude']:
            raise DataError('GEOSEARCH member and longitude or latitude cant be set together')
        pieces.extend([b'FROMMEMBER', kwargs['member']])
    if kwargs['longitude'] is not None and kwargs['latitude'] is not None:
        pieces.extend([b'FROMLONLAT', kwargs['longitude'], kwargs['latitude']])
    if kwargs['radius'] is None:
        if kwargs['width'] is None or kwargs['height'] is None:
            raise DataError('GEOSEARCH must have radius or width and height')
    if kwargs['unit'] is None:
        raise DataError('GEOSEARCH must have unit')
    if kwargs['unit'].lower() not in ('m', 'km', 'mi', 'ft'):
        raise DataError('GEOSEARCH invalid unit')
    if kwargs['radius']:
        if kwargs['width'] or kwargs['height']:
            raise DataError('GEOSEARCH radius and width or height cant be set together')
        pieces.extend([b'BYRADIUS', kwargs['radius'], kwargs['unit']])
    if kwargs['width'] and kwargs['height']:
        pieces.extend([b'BYBOX', kwargs['width'], kwargs['height'], kwargs['unit']])
    if kwargs['sort']:
        if kwargs['sort'].upper() == 'ASC':
            pieces.append(b'ASC')
        elif kwargs['sort'].upper() == 'DESC':
            pieces.append(b'DESC')
        else:
            raise DataError('GEOSEARCH invalid sort')
    if kwargs['count']:
        pieces.extend([b'COUNT', kwargs['count']])
        if kwargs['any']:
            pieces.append(b'ANY')
    elif kwargs['any']:
        raise DataError("GEOSEARCH ``any`` can't be provided without count")
    for arg_name, byte_repr in (('withdist', b'WITHDIST'), ('withcoord', b'WITHCOORD'), ('withhash', b'WITHHASH'), ('store_dist', b'STOREDIST')):
        if kwargs[arg_name]:
            pieces.append(byte_repr)
    return self.execute_command(command, *pieces, **kwargs)