from __future__ import annotations
from collections.abc import Sequence, Set
from typing import Any, Iterable, Union
from referencing import Anchor, Registry, Resource, Specification, exceptions
from referencing._attrs import frozen
from referencing._core import (
from referencing.typing import URI, Anchor as AnchorType, Mapping
def _legacy_dollar_id(contents: Schema) -> URI | None:
    if isinstance(contents, bool) or '$ref' in contents:
        return
    id = contents.get('$id')
    if id is not None and (not id.startswith('#')):
        return id