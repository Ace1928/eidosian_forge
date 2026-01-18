import abc
import sys
import traceback
import warnings
from io import StringIO
from decorator import decorator
from traitlets.config.configurable import Configurable
from .getipython import get_ipython
from ..utils.sentinel import Sentinel
from ..utils.dir2 import get_real_method
from ..lib import pretty
from traitlets import (
from typing import Any
class PNGFormatter(BaseFormatter):
    """A PNG formatter.

    To define the callables that compute the PNG representation of your
    objects, define a :meth:`_repr_png_` method or use the :meth:`for_type`
    or :meth:`for_type_by_name` methods to register functions that handle
    this.

    The return value of this formatter should be raw PNG data, *not*
    base64 encoded.
    """
    format_type = Unicode('image/png')
    print_method = ObjectName('_repr_png_')
    _return_type = (bytes, str)