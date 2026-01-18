from __future__ import annotations
from collections.abc import Sequence, Set
from typing import Any, Iterable, Union
from referencing import Anchor, Registry, Resource, Specification, exceptions
from referencing._attrs import frozen
from referencing._core import (
from referencing.typing import URI, Anchor as AnchorType, Mapping
@frozen
class DynamicAnchor:
    """
    Dynamic anchors, introduced in draft 2020.
    """
    name: str
    resource: SchemaResource

    def resolve(self, resolver: _Resolver[Schema]) -> _Resolved[Schema]:
        """
        Resolve this anchor dynamically.
        """
        last = self.resource
        for uri, registry in resolver.dynamic_scope():
            try:
                anchor = registry.anchor(uri, self.name).value
            except exceptions.NoSuchAnchor:
                continue
            if isinstance(anchor, DynamicAnchor):
                last = anchor.resource
        return _Resolved(contents=last.contents, resolver=resolver.in_subresource(last))