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
def _find_node_path(self, stacktrace):
    import os.path
    last_traceback = stacktrace
    nodes = []
    while hasattr(stacktrace, 'tb_frame'):
        frame = stacktrace.tb_frame
        node = frame.f_locals.get('self')
        if isinstance(node, Nodes.Node):
            code = frame.f_code
            method_name = code.co_name
            pos = (os.path.basename(code.co_filename), frame.f_lineno)
            nodes.append((node, method_name, pos))
            last_traceback = stacktrace
        stacktrace = stacktrace.tb_next
    return (last_traceback, nodes)