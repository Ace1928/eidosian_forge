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
class TransformWrapper(Transform):
    """
    A helper class that holds a single child transform and acts
    equivalently to it.

    This is useful if a node of the transform tree must be replaced at
    run time with a transform of a different type.  This class allows
    that replacement to correctly trigger invalidation.

    `TransformWrapper` instances must have the same input and output dimensions
    during their entire lifetime, so the child transform may only be replaced
    with another child transform of the same dimensions.
    """
    pass_through = True

    def __init__(self, child):
        """
        *child*: A `Transform` instance.  This child may later
        be replaced with :meth:`set`.
        """
        _api.check_isinstance(Transform, child=child)
        super().__init__()
        self.set(child)

    def __eq__(self, other):
        return self._child.__eq__(other)
    __str__ = _make_str_method('_child')

    def frozen(self):
        return self._child.frozen()

    def set(self, child):
        """
        Replace the current child of this transform with another one.

        The new child must have the same number of input and output
        dimensions as the current child.
        """
        if hasattr(self, '_child'):
            self.invalidate()
            new_dims = (child.input_dims, child.output_dims)
            old_dims = (self._child.input_dims, self._child.output_dims)
            if new_dims != old_dims:
                raise ValueError(f'The input and output dims of the new child {new_dims} do not match those of current child {old_dims}')
            self._child._parents.pop(id(self), None)
        self._child = child
        self.set_children(child)
        self.transform = child.transform
        self.transform_affine = child.transform_affine
        self.transform_non_affine = child.transform_non_affine
        self.transform_path = child.transform_path
        self.transform_path_affine = child.transform_path_affine
        self.transform_path_non_affine = child.transform_path_non_affine
        self.get_affine = child.get_affine
        self.inverted = child.inverted
        self.get_matrix = child.get_matrix
        self._invalid = 0
        self.invalidate()
        self._invalid = 0
    input_dims = property(lambda self: self._child.input_dims)
    output_dims = property(lambda self: self._child.output_dims)
    is_affine = property(lambda self: self._child.is_affine)
    is_separable = property(lambda self: self._child.is_separable)
    has_inverse = property(lambda self: self._child.has_inverse)