import sys
import numpy as np
import six
from six.moves import cStringIO as StringIO
from .compat import optional_dep_ok
def _mini_pretty(obj):
    printer = _MiniPPrinter()
    printer.pretty(obj)
    return printer.getvalue()