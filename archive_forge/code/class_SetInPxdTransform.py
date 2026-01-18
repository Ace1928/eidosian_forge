from __future__ import absolute_import
import itertools
from time import time
from . import Errors
from . import DebugFlags
from . import Options
from .Errors import CompileError, InternalError, AbortError
from . import Naming
class SetInPxdTransform(VisitorTransform):

    def visit_StatNode(self, node):
        if hasattr(node, 'in_pxd'):
            node.in_pxd = True
        self.visitchildren(node)
        return node
    visit_Node = VisitorTransform.recurse_to_children