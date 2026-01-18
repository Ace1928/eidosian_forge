import abc
import itertools
import re
from typing import Callable, Iterable, Iterator, List, Optional, Tuple, TypeVar, Union
from .utils import canonicalize_version
from .version import Version
def _is_not_suffix(segment: str) -> bool:
    return not any((segment.startswith(prefix) for prefix in ('dev', 'a', 'b', 'rc', 'post')))