import abc
import itertools
import re
from typing import Callable, Iterable, Iterator, List, Optional, Tuple, TypeVar, Union
from .utils import canonicalize_version
from .version import Version
def _compare_greater_than(self, prospective: Version, spec_str: str) -> bool:
    spec = Version(spec_str)
    if not prospective > spec:
        return False
    if not spec.is_postrelease and prospective.is_postrelease:
        if Version(prospective.base_version) == Version(spec.base_version):
            return False
    if prospective.local is not None:
        if Version(prospective.base_version) == Version(spec.base_version):
            return False
    return True