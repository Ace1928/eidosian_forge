import collections
import warnings
import weakref
from collections.abc import Mapping
from typing import Any, AnyStr, Optional, OrderedDict, Sequence, TypeVar
from scrapy.exceptions import ScrapyDeprecationWarning
def _normkey(self, key: AnyStr) -> AnyStr:
    return key