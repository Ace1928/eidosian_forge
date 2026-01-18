import collections
import itertools
import re
import warnings
from typing import Callable, Iterator, List, Optional, SupportsInt, Tuple, Union
from ._structures import Infinity, InfinityType, NegativeInfinity, NegativeInfinityType
class LegacyVersion(_BaseVersion):

    def __init__(self, version: str) -> None:
        self._version = str(version)
        self._key = _legacy_cmpkey(self._version)
        warnings.warn('Creating a LegacyVersion has been deprecated and will be removed in the next major release', DeprecationWarning)

    def __str__(self) -> str:
        return self._version

    def __repr__(self) -> str:
        return f"<LegacyVersion('{self}')>"

    @property
    def public(self) -> str:
        return self._version

    @property
    def base_version(self) -> str:
        return self._version

    @property
    def epoch(self) -> int:
        return -1

    @property
    def release(self) -> None:
        return None

    @property
    def pre(self) -> None:
        return None

    @property
    def post(self) -> None:
        return None

    @property
    def dev(self) -> None:
        return None

    @property
    def local(self) -> None:
        return None

    @property
    def is_prerelease(self) -> bool:
        return False

    @property
    def is_postrelease(self) -> bool:
        return False

    @property
    def is_devrelease(self) -> bool:
        return False