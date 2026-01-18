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
def _mod_name_key(typ):
    """Return a (__module__, __name__) tuple for a type.

    Used as key in Formatter.deferred_printers.
    """
    module = getattr(typ, '__module__', None)
    name = getattr(typ, '__name__', None)
    return (module, name)