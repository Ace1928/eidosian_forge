from __future__ import absolute_import
from . import Nodes
from . import ExprNodes
from .Nodes import Node
from .ExprNodes import AtomicExprNode
from .PyrexTypes import c_ptr_type, c_bint_type
def _DISABLED_may_be_none(self):
    if self.expression is not None:
        return self.expression.may_be_none()
    if self.type is not None:
        return self.type.is_pyobject
    return True