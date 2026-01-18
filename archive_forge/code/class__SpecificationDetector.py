from __future__ import annotations
from collections.abc import Iterable, Iterator, Sequence
from enum import Enum
from typing import Any, Callable, ClassVar, Generic, Protocol, TypeVar
from urllib.parse import unquote, urldefrag, urljoin
from attrs import evolve, field
from rpds import HashTrieMap, HashTrieSet, List
from referencing import exceptions
from referencing._attrs import frozen
from referencing.typing import URI, Anchor as AnchorType, D, Mapping, Retrieve
class _SpecificationDetector:

    def __get__(self, instance: Specification[D] | None, cls: type[Specification[D]]) -> Callable[[D], Specification[D]]:
        if instance is None:
            return _detect_or_error
        else:
            return _detect_or_default(instance)