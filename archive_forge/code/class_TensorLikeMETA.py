import contextlib
import sys
from typing import Union, TypeVar, Tuple
import numpy as np
from autograd.numpy.numpy_boxes import ArrayBox
class TensorLikeMETA(type):
    """TensorLike metaclass that defines dunder methods for the ``isinstance`` and ``issubclass``
    checks.

    .. note:: These special dunder methods can only be defined inside a metaclass.
    """

    def __instancecheck__(cls, other):
        """Dunder method used to check if an object is a `TensorLike` instance."""
        return isinstance(other, _TensorLike.__args__) or _is_jax(other) or _is_torch(other) or _is_tensorflow(other)

    def __subclasscheck__(cls, other):
        """Dunder method that checks if a class is a subclass of ``TensorLike``."""
        return issubclass(other, _TensorLike.__args__) or _is_jax(other, subclass=True) or _is_torch(other, subclass=True) or _is_tensorflow(other, subclass=True)