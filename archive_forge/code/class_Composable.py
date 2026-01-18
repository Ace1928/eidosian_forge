import numpy as np
import param
from . import traversal
from .dimension import Dimension, Dimensioned, ViewableElement, ViewableTree
from .ndmapping import NdMapping, UniformNdMapping
class Composable(Layoutable):
    """
    Composable is a mix-in class to allow Dimensioned objects to be
    embedded within Layouts and GridSpaces.
    """

    def __lshift__(self, other):
        """Compose objects into an AdjointLayout"""
        if isinstance(other, (ViewableElement, NdMapping, Empty)):
            return AdjointLayout([self, other])
        elif isinstance(other, AdjointLayout):
            return AdjointLayout(other.data.values() + [self])
        else:
            raise TypeError(f'Cannot append {type(other).__name__} to a AdjointLayout')