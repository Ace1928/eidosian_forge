from pythran.analyses.global_declarations import GlobalDeclarations
from pythran.intrinsic import Intrinsic, Class, UnboundValue
from pythran.passmanager import ModuleAnalysis
from pythran.tables import functions, methods, MODULES
from pythran.unparse import Unparser
from pythran.conversion import demangle
import pythran.metadata as md
from pythran.utils import isnum
from pythran.syntax import PythranSyntaxError
import gast as ast
from copy import deepcopy
from itertools import product
import io
@staticmethod
def access_path(node):
    if isinstance(node, ast.Name):
        return MODULES.get(demangle(node.id), node.id)
    elif isinstance(node, ast.Attribute):
        attr_key = demangle(node.attr)
        value_dict = Aliases.access_path(node.value)
        if attr_key not in value_dict:
            raise PythranSyntaxError("Unsupported attribute '{}' for this object".format(attr_key), node.value)
        return value_dict[attr_key]
    elif isinstance(node, ast.FunctionDef):
        return node.name
    else:
        return node