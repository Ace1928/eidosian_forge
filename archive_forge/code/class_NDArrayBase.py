from contextlib import contextmanager
from ipywidgets import register
from traitlets import Unicode, Set, Undefined, Int, validate
import numpy as np
from ..widgets import DataWidget
from .traits import NDArray
from .serializers import compressed_array_serialization
from inspect import Signature, Parameter
class NDArrayBase(DataWidget):
    """A common base class for NDArray-based widgets
    """

    @property
    def shape(self):
        return self._get_shape()

    @property
    def dtype(self):
        return self._get_dtype()

    def _get_shape(self):
        raise NotImplementedError()

    def _get_dtype(self):
        raise NotImplementedError()