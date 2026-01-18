import numpy as np
import param
from . import traversal
from .dimension import Dimension, Dimensioned, ViewableElement, ViewableTree
from .ndmapping import NdMapping, UniformNdMapping
class Layoutable:
    """
    Layoutable provides a mix-in class to support the
    add operation for creating a layout from the operands.
    """

    def __add__(x, y):
        """Compose objects into a Layout"""
        if any((isinstance(arg, int) for arg in (x, y))):
            raise TypeError(f'unsupported operand type(s) for +: {x.__class__.__name__} and {y.__class__.__name__}. If you are trying to use a reduction like `sum(elements)` to combine a list of elements, we recommend you use `Layout(elements)` (and similarly `Overlay(elements)` for making an overlay from a list) instead.')
        try:
            return Layout([x, y])
        except NotImplementedError:
            return NotImplemented

    def __radd__(self, other):
        return self.__class__.__add__(other, self)