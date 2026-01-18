from __future__ import annotations
import re
import typing as t
import uuid
from urllib.parse import quote
class AnyConverter(BaseConverter):
    """Matches one of the items provided.  Items can either be Python
    identifiers or strings::

        Rule('/<any(about, help, imprint, class, "foo,bar"):page_name>')

    :param map: the :class:`Map`.
    :param items: this function accepts the possible items as positional
                  arguments.

    .. versionchanged:: 2.2
        Value is validated when building a URL.
    """

    def __init__(self, map: Map, *items: str) -> None:
        super().__init__(map)
        self.items = set(items)
        self.regex = f'(?:{'|'.join([re.escape(x) for x in items])})'

    def to_url(self, value: t.Any) -> str:
        if value in self.items:
            return str(value)
        valid_values = ', '.join((f"'{item}'" for item in sorted(self.items)))
        raise ValueError(f"'{value}' is not one of {valid_values}")