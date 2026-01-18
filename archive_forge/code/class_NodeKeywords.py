import collections.abc
import dataclasses
import inspect
from typing import Any
from typing import Callable
from typing import Collection
from typing import final
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import MutableMapping
from typing import NamedTuple
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import warnings
from .._code import getfslineno
from ..compat import ascii_escaped
from ..compat import NOTSET
from ..compat import NotSetType
from _pytest.config import Config
from _pytest.deprecated import check_ispytest
from _pytest.deprecated import MARKED_FIXTURE
from _pytest.outcomes import fail
from _pytest.warning_types import PytestUnknownMarkWarning
@final
class NodeKeywords(MutableMapping[str, Any]):
    __slots__ = ('node', 'parent', '_markers')

    def __init__(self, node: 'Node') -> None:
        self.node = node
        self.parent = node.parent
        self._markers = {node.name: True}

    def __getitem__(self, key: str) -> Any:
        try:
            return self._markers[key]
        except KeyError:
            if self.parent is None:
                raise
            return self.parent.keywords[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._markers[key] = value

    def __contains__(self, key: object) -> bool:
        return key in self._markers or (self.parent is not None and key in self.parent.keywords)

    def update(self, other: Union[Mapping[str, Any], Iterable[Tuple[str, Any]]]=(), **kwds: Any) -> None:
        self._markers.update(other)
        self._markers.update(kwds)

    def __delitem__(self, key: str) -> None:
        raise ValueError('cannot delete key in keywords dict')

    def __iter__(self) -> Iterator[str]:
        yield from self._markers
        if self.parent is not None:
            for keyword in self.parent.keywords:
                if keyword not in self._markers:
                    yield keyword

    def __len__(self) -> int:
        return sum((1 for keyword in self))

    def __repr__(self) -> str:
        return f'<NodeKeywords for node {self.node}>'