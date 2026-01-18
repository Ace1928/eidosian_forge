from __future__ import absolute_import
from . import Nodes
from . import ExprNodes
from .Nodes import Node
from .ExprNodes import AtomicExprNode
from .PyrexTypes import c_ptr_type, c_bint_type

    Simple node that evaluates to 0 or 1 depending on whether we're
    in a nogil context
    