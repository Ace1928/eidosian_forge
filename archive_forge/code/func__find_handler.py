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
def _find_handler(self, match_name, has_kwargs):
    try:
        match_name.encode('ascii')
    except UnicodeEncodeError:
        return None
    call_type = 'general' if has_kwargs else 'simple'
    handler = getattr(self, '_handle_%s_%s' % (call_type, match_name), None)
    if handler is None:
        handler = getattr(self, '_handle_any_%s' % match_name, None)
    return handler