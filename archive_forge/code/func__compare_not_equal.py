import abc
import itertools
import re
from typing import Callable, Iterable, Iterator, List, Optional, Tuple, TypeVar, Union
from .utils import canonicalize_version
from .version import Version
def _compare_not_equal(self, prospective: Version, spec: str) -> bool:
    return not self._compare_equal(prospective, spec)