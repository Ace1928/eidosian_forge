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
def _find_property_decorator(self, node):
    properties = self._properties[-1]
    for decorator_node in node.decorators[::-1]:
        decorator = decorator_node.decorator
        if decorator.is_name and decorator.name == 'property':
            return decorator_node
        elif decorator.is_attribute and decorator.obj.name in properties:
            return decorator_node
    return None