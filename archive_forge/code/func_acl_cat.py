import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def acl_cat(self, category: Union[str, None]=None, **kwargs) -> ResponseT:
    """
        Returns a list of categories or commands within a category.

        If ``category`` is not supplied, returns a list of all categories.
        If ``category`` is supplied, returns a list of all commands within
        that category.

        For more information see https://redis.io/commands/acl-cat
        """
    pieces: list[EncodableT] = [category] if category else []
    return self.execute_command('ACL CAT', *pieces, **kwargs)