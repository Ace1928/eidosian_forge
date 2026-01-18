import re
from cwcwidth import wcswidth, wcwidth
from itertools import chain
from typing import (
from .escseqparse import parse, remove_ansi
from .termformatconstants import (
class FrozenAttributes(Dict[str, Union[int, bool]]):
    """Immutable dictionary class for format string attributes"""

    def __setitem__(self, key: str, value: Union[int, bool]) -> None:
        raise Exception('Cannot change value.')

    def update(self, *args: Any, **kwds: Any) -> None:
        raise Exception('Cannot change value.')

    def extend(self, dictlike: Mapping[str, Union[int, bool]]) -> 'FrozenAttributes':
        return FrozenAttributes(chain(self.items(), dictlike.items()))

    def remove(self, *keys: str) -> 'FrozenAttributes':
        return FrozenAttributes(((k, v) for k, v in self.items() if k not in keys))