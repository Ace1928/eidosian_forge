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
def _create_cimport_from_import(self, node_pos, module_name, level, imported_names):
    if module_name == u'cython.cimports' or module_name.startswith(u'cython.cimports.'):
        module_name = EncodedString(module_name[len(u'cython.cimports.'):])
    if module_name:
        return Nodes.FromCImportStatNode(node_pos, module_name=module_name, relative_level=level, imported_names=imported_names)
    else:
        return [Nodes.CImportStatNode(pos, module_name=dotted_name, as_name=as_name, is_absolute=level == 0) for pos, dotted_name, as_name in imported_names]