from __future__ import annotations
from collections.abc import Sequence, Set
from typing import Any, Iterable, Union
from referencing import Anchor, Registry, Resource, Specification, exceptions
from referencing._attrs import frozen
from referencing._core import (
from referencing.typing import URI, Anchor as AnchorType, Mapping
def _legacy_id(contents: ObjectSchema) -> URI | None:
    if '$ref' in contents:
        return
    id = contents.get('id')
    if id is not None and (not id.startswith('#')):
        return id