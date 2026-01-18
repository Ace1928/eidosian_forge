from __future__ import absolute_import, print_function
import sys
import inspect
from . import TypeSlots
from . import Builtin
from . import Nodes
from . import ExprNodes
from . import Errors
from . import DebugFlags
from . import Future
import cython
@cython.final
def _process_children(self, parent, attrs=None, exclude=None):
    result = self._visitchildren(parent, attrs, exclude)
    for attr, newnode in result.items():
        if type(newnode) is list:
            newnode = self._flatten_list(newnode)
        setattr(parent, attr, newnode)
    return result