from __future__ import absolute_import
import cython
from . import Builtin
from . import ExprNodes
from . import Nodes
from . import Options
from . import PyrexTypes
from .Visitor import TreeVisitor, CythonTransform
from .Errors import error, warning, InternalError
class NameReference(object):

    def __init__(self, node, entry):
        if node.cf_state is None:
            node.cf_state = set()
        self.node = node
        self.entry = entry
        self.pos = node.pos

    def __repr__(self):
        return '%s(entry=%r)' % (self.__class__.__name__, self.entry)