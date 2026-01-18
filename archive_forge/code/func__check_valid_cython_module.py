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
def _check_valid_cython_module(self, pos, module_name):
    if not module_name.startswith('cython.'):
        return
    submodule = module_name.split('.', 2)[1]
    if submodule in self.valid_cython_submodules:
        return
    extra = ''
    hints = [line.split() for line in '                imp                  cimports\n                cimp                 cimports\n                para                 parallel\n                parra                parallel\n                dataclass            dataclasses\n            '.splitlines()[:-1]]
    for wrong, correct in hints:
        if module_name.startswith('cython.' + wrong):
            extra = "Did you mean 'cython.%s' ?" % correct
            break
    if not extra:
        is_simple_cython_name = submodule in Options.directive_types
        if not is_simple_cython_name and (not submodule.startswith('_')):
            from .. import Shadow
            is_simple_cython_name = hasattr(Shadow, submodule)
        if is_simple_cython_name:
            extra = "Instead, use 'import cython' and then 'cython.%s'." % submodule
    error(pos, "'%s' is not a valid cython.* module%s%s" % (module_name, '. ' if extra else '', extra))