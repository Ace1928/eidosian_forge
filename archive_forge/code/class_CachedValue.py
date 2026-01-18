from __future__ import annotations
import abc
import pickle
import time
from typing import Any
from typing import Callable
from typing import cast
from typing import Mapping
from typing import NamedTuple
from typing import Optional
from typing import Sequence
from typing import Union
from ..util.typing import Self
class CachedValue(NamedTuple):
    """Represent a value stored in the cache.

    :class:`.CachedValue` is a two-tuple of
    ``(payload, metadata)``, where ``metadata``
    is dogpile.cache's tracking information (
    currently the creation time).

    """
    payload: ValuePayload
    'the actual cached value.'
    metadata: MetaDataType
    'Metadata dictionary for the cached value.\n\n    Prefer using accessors such as :attr:`.CachedValue.cached_time` rather\n    than accessing this mapping directly.\n\n    '

    @property
    def cached_time(self) -> float:
        """The epoch (floating point time value) stored when this payload was
        cached.

        .. versionadded:: 1.3

        """
        return cast(float, self.metadata['ct'])

    @property
    def age(self) -> float:
        """Returns the elapsed time in seconds as a `float` since the insertion
        of the value in the cache.

        This value is computed **dynamically** by subtracting the cached
        floating point epoch value from the value of ``time.time()``.

        .. versionadded:: 1.3

        """
        return time.time() - self.cached_time