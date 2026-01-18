from __future__ import absolute_import
import cython
import copy
import hashlib
import sys
from . import PyrexTypes
from . import Naming
from . import ExprNodes
from . import Nodes
from . import Options
from . import Builtin
from . import Errors
from .Visitor import VisitorTransform, TreeVisitor
from .Visitor import CythonTransform, EnvTransform, ScopeTrackingTransform
from .UtilNodes import LetNode, LetRefNode
from .TreeFragment import TreeFragment
from .StringEncoding import EncodedString, _unicode
from .Errors import error, warning, CompileError, InternalError
from .Code import UtilityCode
class PxdPostParse(CythonTransform, SkipDeclarations):
    """
    Basic interpretation/validity checking that should only be
    done on pxd trees.

    A lot of this checking currently happens in the parser; but
    what is listed below happens here.

    - "def" functions are let through only if they fill the
    getbuffer/releasebuffer slots

    - cdef functions are let through only if they are on the
    top level and are declared "inline"
    """
    ERR_INLINE_ONLY = "function definition in pxd file must be declared 'cdef inline'"
    ERR_NOGO_WITH_INLINE = "inline function definition in pxd file cannot be '%s'"

    def __call__(self, node):
        self.scope_type = 'pxd'
        return super(PxdPostParse, self).__call__(node)

    def visit_CClassDefNode(self, node):
        old = self.scope_type
        self.scope_type = 'cclass'
        self.visitchildren(node)
        self.scope_type = old
        return node

    def visit_FuncDefNode(self, node):
        err = self.ERR_INLINE_ONLY
        if isinstance(node, Nodes.DefNode) and self.scope_type == 'cclass' and (node.name in ('__getbuffer__', '__releasebuffer__')):
            err = None
        if isinstance(node, Nodes.CFuncDefNode):
            if u'inline' in node.modifiers and self.scope_type in ('pxd', 'cclass'):
                node.inline_in_pxd = True
                if node.visibility != 'private':
                    err = self.ERR_NOGO_WITH_INLINE % node.visibility
                elif node.api:
                    err = self.ERR_NOGO_WITH_INLINE % 'api'
                else:
                    err = None
            else:
                err = self.ERR_INLINE_ONLY
        if err:
            self.context.nonfatal_error(PostParseError(node.pos, err))
            return None
        else:
            return node