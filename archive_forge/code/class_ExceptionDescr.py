from __future__ import absolute_import
import cython
from . import Builtin
from . import ExprNodes
from . import Nodes
from . import Options
from . import PyrexTypes
from .Visitor import TreeVisitor, CythonTransform
from .Errors import error, warning, InternalError
class ExceptionDescr(object):
    """Exception handling helper.

    entry_point   ControlBlock Exception handling entry point
    finally_enter ControlBlock Normal finally clause entry point
    finally_exit  ControlBlock Normal finally clause exit point
    """

    def __init__(self, entry_point, finally_enter=None, finally_exit=None):
        self.entry_point = entry_point
        self.finally_enter = finally_enter
        self.finally_exit = finally_exit