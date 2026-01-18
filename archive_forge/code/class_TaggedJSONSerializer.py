from __future__ import annotations
import typing as t
from base64 import b64decode
from base64 import b64encode
from datetime import datetime
from uuid import UUID
from markupsafe import Markup
from werkzeug.http import http_date
from werkzeug.http import parse_date
from ..json import dumps
from ..json import loads
class TaggedJSONSerializer:
    """Serializer that uses a tag system to compactly represent objects that
    are not JSON types. Passed as the intermediate serializer to
    :class:`itsdangerous.Serializer`.

    The following extra types are supported:

    * :class:`dict`
    * :class:`tuple`
    * :class:`bytes`
    * :class:`~markupsafe.Markup`
    * :class:`~uuid.UUID`
    * :class:`~datetime.datetime`
    """
    __slots__ = ('tags', 'order')
    default_tags = [TagDict, PassDict, TagTuple, PassList, TagBytes, TagMarkup, TagUUID, TagDateTime]

    def __init__(self) -> None:
        self.tags: dict[str, JSONTag] = {}
        self.order: list[JSONTag] = []
        for cls in self.default_tags:
            self.register(cls)

    def register(self, tag_class: type[JSONTag], force: bool=False, index: int | None=None) -> None:
        """Register a new tag with this serializer.

        :param tag_class: tag class to register. Will be instantiated with this
            serializer instance.
        :param force: overwrite an existing tag. If false (default), a
            :exc:`KeyError` is raised.
        :param index: index to insert the new tag in the tag order. Useful when
            the new tag is a special case of an existing tag. If ``None``
            (default), the tag is appended to the end of the order.

        :raise KeyError: if the tag key is already registered and ``force`` is
            not true.
        """
        tag = tag_class(self)
        key = tag.key
        if key:
            if not force and key in self.tags:
                raise KeyError(f"Tag '{key}' is already registered.")
            self.tags[key] = tag
        if index is None:
            self.order.append(tag)
        else:
            self.order.insert(index, tag)

    def tag(self, value: t.Any) -> t.Any:
        """Convert a value to a tagged representation if necessary."""
        for tag in self.order:
            if tag.check(value):
                return tag.tag(value)
        return value

    def untag(self, value: dict[str, t.Any]) -> t.Any:
        """Convert a tagged representation back to the original type."""
        if len(value) != 1:
            return value
        key = next(iter(value))
        if key not in self.tags:
            return value
        return self.tags[key].to_python(value[key])

    def _untag_scan(self, value: t.Any) -> t.Any:
        if isinstance(value, dict):
            value = {k: self._untag_scan(v) for k, v in value.items()}
            value = self.untag(value)
        elif isinstance(value, list):
            value = [self._untag_scan(item) for item in value]
        return value

    def dumps(self, value: t.Any) -> str:
        """Tag the value and dump it to a compact JSON string."""
        return dumps(self.tag(value), separators=(',', ':'))

    def loads(self, value: str) -> t.Any:
        """Load data from a JSON string and deserialized any tagged objects."""
        return self._untag_scan(loads(value))