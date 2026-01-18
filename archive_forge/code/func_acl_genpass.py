import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def acl_genpass(self, bits: Union[int, None]=None, **kwargs) -> ResponseT:
    """Generate a random password value.
        If ``bits`` is supplied then use this number of bits, rounded to
        the next multiple of 4.
        See: https://redis.io/commands/acl-genpass
        """
    pieces = []
    if bits is not None:
        try:
            b = int(bits)
            if b < 0 or b > 4096:
                raise ValueError
            pieces.append(b)
        except ValueError:
            raise DataError('genpass optionally accepts a bits argument, between 0 and 4096.')
    return self.execute_command('ACL GENPASS', *pieces, **kwargs)