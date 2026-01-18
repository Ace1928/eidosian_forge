from __future__ import annotations
import typing
from typing import Any
from typing import Dict
from typing import Tuple
from typing import Union
from ..util.typing import Literal
class prefix_anon_map(Dict[str, str]):
    """A map that creates new keys for missing key access.

    Considers keys of the form "<ident> <name>" to produce
    new symbols "<name>_<index>", where "index" is an incrementing integer
    corresponding to <name>.

    Inlines the approach taken by :class:`sqlalchemy.util.PopulateDict` which
    is otherwise usually used for this type of operation.

    """

    def __missing__(self, key: str) -> str:
        ident, derived = key.split(' ', 1)
        anonymous_counter = self.get(derived, 1)
        self[derived] = anonymous_counter + 1
        value = f'{derived}_{anonymous_counter}'
        self[key] = value
        return value