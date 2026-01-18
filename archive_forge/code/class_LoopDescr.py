from __future__ import absolute_import
import cython
from . import Builtin
from . import ExprNodes
from . import Nodes
from . import Options
from . import PyrexTypes
from .Visitor import TreeVisitor, CythonTransform
from .Errors import error, warning, InternalError
class LoopDescr(object):

    def __init__(self, next_block, loop_block):
        self.next_block = next_block
        self.loop_block = loop_block
        self.exceptions = []