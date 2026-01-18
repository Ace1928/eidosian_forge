from __future__ import annotations
from collections.abc import Sequence, Set
from typing import Any, Iterable, Union
from referencing import Anchor, Registry, Resource, Specification, exceptions
from referencing._attrs import frozen
from referencing._core import (
from referencing.typing import URI, Anchor as AnchorType, Mapping
def _subresources_of_with_crazy_aP_items_dependencies(in_value: Set[str]=frozenset(), in_subvalues: Set[str]=frozenset(), in_subarray: Set[str]=frozenset()):
    """
    Specifically handle even older drafts where there are some funky keywords.
    """

    def subresources_of(contents: ObjectSchema) -> Iterable[ObjectSchema]:
        for each in in_value:
            if each in contents:
                yield contents[each]
        for each in in_subarray:
            if each in contents:
                yield from contents[each]
        for each in in_subvalues:
            if each in contents:
                yield from contents[each].values()
        items = contents.get('items')
        if items is not None:
            if isinstance(items, Sequence):
                yield from items
            else:
                yield items
        dependencies = contents.get('dependencies')
        if dependencies is not None:
            values = iter(dependencies.values())
            value = next(values, None)
            if isinstance(value, Mapping):
                yield value
                yield from values
        for each in ('additionalItems', 'additionalProperties'):
            value = contents.get(each)
            if isinstance(value, Mapping):
                yield value
    return subresources_of