from __future__ import annotations
from contextlib import suppress
from typing import TYPE_CHECKING
from warnings import warn
import numpy as np
from .._utils import to_rgba
from .._utils.registry import RegistryHierarchyMeta
from ..exceptions import PlotnineError, deprecated_themeable_name
from .elements import element_blank
from .elements.element_base import element_base
class MixinSequenceOfValues(themeable):
    """
    Make themeable also accept a sequence to values

    This makes it possible to apply a different style value similar artists.

    e.g.

        theme(axis_text_x=element_text(color=("red", "green", "blue")))

    The number of values in the list must match the number of objects
    targetted by the themeable..
    """

    def set(self, artists: Sequence[Artist], props: Optional[dict[str, Any]]=None):
        if props is None:
            props = self.properties
        n = len(artists)
        sequence_props = {}
        for name, value in props.items():
            if isinstance(value, (list, tuple, np.ndarray)) and len(value) == n:
                sequence_props[name] = value
        for key in sequence_props:
            del props[key]
        for a in artists:
            a.set(**props)
        for name, values in sequence_props.items():
            for a, value in zip(artists, values):
                a.set(**{name: value})