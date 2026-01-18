from __future__ import absolute_import
import cython
from . import Builtin
from . import ExprNodes
from . import Nodes
from . import Options
from . import PyrexTypes
from .Visitor import TreeVisitor, CythonTransform
from .Errors import error, warning, InternalError
def _delete_privates(self, node, exclude=None):
    for private_node in node.assigned_nodes:
        if not exclude or private_node.entry is not exclude:
            self.flow.mark_deletion(private_node, private_node.entry)