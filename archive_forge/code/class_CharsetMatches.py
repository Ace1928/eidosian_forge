import warnings
from collections import Counter
from encodings.aliases import aliases
from hashlib import sha256
from json import dumps
from re import sub
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
from .constant import NOT_PRINTABLE_PATTERN, TOO_BIG_SEQUENCE
from .md import mess_ratio
from .utils import iana_name, is_multi_byte_encoding, unicode_range
class CharsetMatches:
    """
    Container with every CharsetMatch items ordered by default from most probable to the less one.
    Act like a list(iterable) but does not implements all related methods.
    """

    def __init__(self, results: List[CharsetMatch]=None):
        self._results = sorted(results) if results else []

    def __iter__(self) -> Iterator[CharsetMatch]:
        yield from self._results

    def __getitem__(self, item: Union[int, str]) -> CharsetMatch:
        """
        Retrieve a single item either by its position or encoding name (alias may be used here).
        Raise KeyError upon invalid index or encoding not present in results.
        """
        if isinstance(item, int):
            return self._results[item]
        if isinstance(item, str):
            item = iana_name(item, False)
            for result in self._results:
                if item in result.could_be_from_charset:
                    return result
        raise KeyError

    def __len__(self) -> int:
        return len(self._results)

    def __bool__(self) -> bool:
        return len(self._results) > 0

    def append(self, item: CharsetMatch) -> None:
        """
        Insert a single match. Will be inserted accordingly to preserve sort.
        Can be inserted as a submatch.
        """
        if not isinstance(item, CharsetMatch):
            raise ValueError("Cannot append instance '{}' to CharsetMatches".format(str(item.__class__)))
        if len(item.raw) <= TOO_BIG_SEQUENCE:
            for match in self._results:
                if match.fingerprint == item.fingerprint and match.chaos == item.chaos:
                    match.add_submatch(item)
                    return
        self._results.append(item)
        self._results = sorted(self._results)

    def best(self) -> Optional['CharsetMatch']:
        """
        Simply return the first match. Strict equivalent to matches[0].
        """
        if not self._results:
            return None
        return self._results[0]

    def first(self) -> Optional['CharsetMatch']:
        """
        Redundant method, call the method best(). Kept for BC reasons.
        """
        return self.best()