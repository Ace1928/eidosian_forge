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
class JavascriptFormatter(BaseFormatter):
    """A Javascript formatter.

    To define the callables that compute the Javascript representation of
    your objects, define a :meth:`_repr_javascript_` method or use the
    :meth:`for_type` or :meth:`for_type_by_name` methods to register functions
    that handle this.

    The return value of this formatter should be valid Javascript code and
    should *not* be enclosed in ```<script>``` tags.
    """
    format_type = Unicode('application/javascript')
    print_method = ObjectName('_repr_javascript_')