from __future__ import annotations
from collections.abc import Callable, Mapping
from typing import Any, TypeVar, final, overload
from ._exceptions import TypedAttributeLookupError

        extra(attribute, default=undefined)

        Return the value of the given typed extra attribute.

        :param attribute: the attribute (member of a :class:`~TypedAttributeSet`) to
            look for
        :param default: the value that should be returned if no value is found for the
            attribute
        :raises ~anyio.TypedAttributeLookupError: if the search failed and no default
            value was given

        