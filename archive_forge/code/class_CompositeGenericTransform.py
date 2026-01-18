import copy
import functools
import textwrap
import weakref
import math
import numpy as np
from numpy.linalg import inv
from matplotlib import _api
from matplotlib._path import (
from .path import Path
class CompositeGenericTransform(Transform):
    """
    A composite transform formed by applying transform *a* then
    transform *b*.

    This "generic" version can handle any two arbitrary
    transformations.
    """
    pass_through = True

    def __init__(self, a, b, **kwargs):
        """
        Create a new composite transform that is the result of
        applying transform *a* then transform *b*.

        You will generally not call this constructor directly but write ``a +
        b`` instead, which will automatically choose the best kind of composite
        transform instance to create.
        """
        if a.output_dims != b.input_dims:
            raise ValueError("The output dimension of 'a' must be equal to the input dimensions of 'b'")
        self.input_dims = a.input_dims
        self.output_dims = b.output_dims
        super().__init__(**kwargs)
        self._a = a
        self._b = b
        self.set_children(a, b)

    def frozen(self):
        self._invalid = 0
        frozen = composite_transform_factory(self._a.frozen(), self._b.frozen())
        if not isinstance(frozen, CompositeGenericTransform):
            return frozen.frozen()
        return frozen

    def _invalidate_internal(self, level, invalidating_node):
        if invalidating_node is self._a and (not self._b.is_affine):
            level = Transform._INVALID_FULL
        super()._invalidate_internal(level, invalidating_node)

    def __eq__(self, other):
        if isinstance(other, (CompositeGenericTransform, CompositeAffine2D)):
            return self is other or (self._a == other._a and self._b == other._b)
        else:
            return False

    def _iter_break_from_left_to_right(self):
        for left, right in self._a._iter_break_from_left_to_right():
            yield (left, right + self._b)
        for left, right in self._b._iter_break_from_left_to_right():
            yield (self._a + left, right)
    depth = property(lambda self: self._a.depth + self._b.depth)
    is_affine = property(lambda self: self._a.is_affine and self._b.is_affine)
    is_separable = property(lambda self: self._a.is_separable and self._b.is_separable)
    has_inverse = property(lambda self: self._a.has_inverse and self._b.has_inverse)
    __str__ = _make_str_method('_a', '_b')

    @_api.rename_parameter('3.8', 'points', 'values')
    def transform_affine(self, values):
        return self.get_affine().transform(values)

    @_api.rename_parameter('3.8', 'points', 'values')
    def transform_non_affine(self, values):
        if self._a.is_affine and self._b.is_affine:
            return values
        elif not self._a.is_affine and self._b.is_affine:
            return self._a.transform_non_affine(values)
        else:
            return self._b.transform_non_affine(self._a.transform(values))

    def transform_path_non_affine(self, path):
        if self._a.is_affine and self._b.is_affine:
            return path
        elif not self._a.is_affine and self._b.is_affine:
            return self._a.transform_path_non_affine(path)
        else:
            return self._b.transform_path_non_affine(self._a.transform_path(path))

    def get_affine(self):
        if not self._b.is_affine:
            return self._b.get_affine()
        else:
            return Affine2D(np.dot(self._b.get_affine().get_matrix(), self._a.get_affine().get_matrix()))

    def inverted(self):
        return CompositeGenericTransform(self._b.inverted(), self._a.inverted())