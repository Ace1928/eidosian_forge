from __future__ import annotations
from collections.abc import Sequence, Set
from typing import Any, Iterable, Union
from referencing import Anchor, Registry, Resource, Specification, exceptions
from referencing._attrs import frozen
from referencing._core import (
from referencing.typing import URI, Anchor as AnchorType, Mapping
def _anchor_2019(specification: Specification[Schema], contents: Schema) -> Iterable[Anchor[Schema]]:
    if isinstance(contents, bool):
        return []
    anchor = contents.get('$anchor')
    if anchor is None:
        return []
    return [Anchor(name=anchor, resource=specification.create_resource(contents))]