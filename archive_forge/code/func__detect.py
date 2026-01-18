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
def _detect(contents: D) -> Specification[D]:
    if not isinstance(contents, Mapping):
        return default
    jsonschema_dialect_id = contents.get('$schema')
    if jsonschema_dialect_id is None:
        return default
    from referencing.jsonschema import specification_with
    return specification_with(jsonschema_dialect_id, default=default)