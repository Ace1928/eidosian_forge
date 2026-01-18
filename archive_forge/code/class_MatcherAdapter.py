import ast
import dataclasses
import enum
import re
import types
from typing import Callable
from typing import Iterator
from typing import Mapping
from typing import NoReturn
from typing import Optional
from typing import Sequence
class MatcherAdapter(Mapping[str, bool]):
    """Adapts a matcher function to a locals mapping as required by eval()."""

    def __init__(self, matcher: Callable[[str], bool]) -> None:
        self.matcher = matcher

    def __getitem__(self, key: str) -> bool:
        return self.matcher(key[len(IDENT_PREFIX):])

    def __iter__(self) -> Iterator[str]:
        raise NotImplementedError()

    def __len__(self) -> int:
        raise NotImplementedError()