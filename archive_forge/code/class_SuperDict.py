from __future__ import annotations
import typing as t
from .exceptions import ListparserError
from .xml_handler import XMLHandler
class SuperDict(t.Dict[str, t.Any]):
    """
    SuperDict is a dictionary object with keys posing as instance attributes.

    ..  code-block:: pycon

        >>> i = SuperDict()
        >>> i.one = 1
        >>> i
        {'one': 1}

    """

    def __getattribute__(self, name: str) -> t.Any:
        if name in self:
            return self[name]
        else:
            return dict.__getattribute__(self, name)