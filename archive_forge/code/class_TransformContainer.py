import functools
import os
import copy
import warnings
import types
from typing import Sequence
import pennylane as qml
from pennylane.typing import ResultBatch
class TransformContainer:
    """Class to store a quantum transform with its ``args``, ``kwargs`` and classical co-transforms.  Use
    :func:`~.pennylane.transform`.

    .. warning::

        This class is developer-facing and should not be used directly. Instead, use
        :func:`qml.transform <pennylane.transform>` if you would like to make a custom
        transform.

    .. seealso:: :func:`~.pennylane.transform`
    """

    def __init__(self, transform, args=None, kwargs=None, classical_cotransform=None, is_informative=False, final_transform=False, use_argnum=False):
        self._transform = transform
        self._args = args or []
        self._kwargs = kwargs or {}
        self._classical_cotransform = classical_cotransform
        self._is_informative = is_informative
        self._final_transform = is_informative or final_transform
        self._use_argnum = use_argnum

    def __repr__(self):
        return f'<{self._transform.__name__}({self._args}, {self._kwargs})>'

    def __iter__(self):
        return iter((self._transform, self._args, self._kwargs, self._classical_cotransform, self._is_informative, self.final_transform))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TransformContainer):
            return False
        return self.args == other.args and self.transform == other.transform and (self.kwargs == other.kwargs) and (self.classical_cotransform == other.classical_cotransform) and (self.is_informative == other.is_informative) and (self.final_transform == other.final_transform)

    @property
    def transform(self):
        """The stored quantum transform."""
        return self._transform

    @property
    def args(self):
        """The stored quantum transform's ``args``."""
        return self._args

    @property
    def kwargs(self):
        """The stored quantum transform's ``kwargs``."""
        return self._kwargs

    @property
    def classical_cotransform(self):
        """The stored quantum transform's classical co-transform."""
        return self._classical_cotransform

    @property
    def is_informative(self):
        """``True`` if the transform is informative."""
        return self._is_informative

    @property
    def final_transform(self):
        """``True`` if the transform needs to be executed"""
        return self._final_transform