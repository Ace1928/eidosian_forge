from functools import reduce
import numpy as np
import param
from .dimension import Dimension, Dimensioned, ViewableElement, ViewableTree
from .layout import AdjointLayout, Composable, Layout, Layoutable
from .ndmapping import UniformNdMapping
from .util import dimensioned_streams, sanitize_identifier, unique_array
class Overlayable:
    """
    Overlayable provides a mix-in class to support the
    mul operation for overlaying multiple elements.
    """

    def __mul__(self, other):
        """Overlay object with other object."""
        from .spaces import DynamicMap
        if isinstance(other, DynamicMap):
            from .spaces import Callable

            def dynamic_mul(*args, **kwargs):
                element = other[args]
                return self * element
            callback = Callable(dynamic_mul, inputs=[self, other])
            callback._is_overlay = True
            return other.clone(shared_data=False, callback=callback, streams=dimensioned_streams(other))
        else:
            if isinstance(self, Overlay):
                if not isinstance(other, ViewableElement):
                    return NotImplemented
            elif isinstance(other, UniformNdMapping) and (not isinstance(other, CompositeOverlay)):
                items = [(k, self * v) for k, v in other.items()]
                return other.clone(items)
            elif isinstance(other, (AdjointLayout, ViewableTree)) and (not isinstance(other, Overlay)):
                return NotImplemented
            try:
                return Overlay([self, other])
            except NotImplementedError:
                return NotImplemented