from __future__ import annotations
from collections.abc import Iterable
from dataclasses import make_dataclass
class DataBinMeta(type):
    """Metaclass for :class:`DataBin` that adds the shape to the type name.

    This is so that the class has a custom repr with DataBin<*shape> notation.
    """

    def __repr__(cls):
        name = cls.__name__
        if cls._SHAPE is None:
            return name
        shape = ','.join(map(str, cls._SHAPE))
        return f'{name}<{shape}>'