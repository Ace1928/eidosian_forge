import collections
import warnings
import weakref
from collections.abc import Mapping
from typing import Any, AnyStr, Optional, OrderedDict, Sequence, TypeVar
from scrapy.exceptions import ScrapyDeprecationWarning
class LocalCache(OrderedDict[_KT, _VT]):
    """Dictionary with a finite number of keys.

    Older items expires first.
    """

    def __init__(self, limit: Optional[int]=None):
        super().__init__()
        self.limit: Optional[int] = limit

    def __setitem__(self, key: _KT, value: _VT) -> None:
        if self.limit:
            while len(self) >= self.limit:
                self.popitem(last=False)
        super().__setitem__(key, value)