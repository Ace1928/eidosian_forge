from contextlib import contextmanager
from ipywidgets import register
from traitlets import Unicode, Set, Undefined, Int, validate
import numpy as np
from ..widgets import DataWidget
from .traits import NDArray
from .serializers import compressed_array_serialization
from inspect import Signature, Parameter
@register
class NDArraySource(NDArrayBase):
    """Base class for widgets that supplies an ndarray in the front-end only.
    """
    pass