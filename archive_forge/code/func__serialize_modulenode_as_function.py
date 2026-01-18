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
def _serialize_modulenode_as_function(self, node, attrs):
    self.tb.start('Function', attrs=attrs)
    self.tb.start('Locals')
    self.serialize_local_variables(node.scope.entries)
    self.tb.end('Locals')
    self.tb.start('Arguments')
    self.tb.end('Arguments')
    self.tb.start('StepIntoFunctions')
    self.register_stepinto = True
    self.visitchildren(node)
    self.register_stepinto = False
    self.tb.end('StepIntoFunctions')
    self.tb.end('Function')