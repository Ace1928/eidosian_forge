import os
import sys
import typing
from contextlib import contextmanager
from collections.abc import Iterable
from IPython import get_ipython
from traitlets import (
from json import loads as jsonloads, dumps as jsondumps
from .. import comm
from base64 import standard_b64encode
from .utils import deprecation, _get_frame
from .._version import __protocol_version__, __control_protocol_version__, __jupyter_widgets_base_version__
import inspect
@contextmanager
def _lock_property(self, **properties):
    """Lock a property-value pair.

        The value should be the JSON state of the property.

        NOTE: This, in addition to the single lock for all state changes, is
        flawed.  In the future we may want to look into buffering state changes
        back to the front-end."""
    self._property_lock = properties
    try:
        yield
    finally:
        self._property_lock = {}