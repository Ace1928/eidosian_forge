from pythran.analyses.aliases import Aliases
from pythran.analyses.intrinsics import Intrinsics
from pythran.analyses.global_declarations import GlobalDeclarations
from pythran.passmanager import ModuleAnalysis
from pythran.tables import MODULES
from pythran.graph import DiGraph
import pythran.intrinsic as intrinsic
import gast as ast
from functools import reduce
class FunctionEffect(object):

    def __init__(self, node):
        self.func = node
        if isinstance(node, ast.FunctionDef):
            self.global_effect = False
        elif isinstance(node, intrinsic.Intrinsic):
            self.global_effect = node.global_effects
        elif isinstance(node, ast.alias):
            self.global_effect = False
        elif isinstance(node, str):
            self.global_effect = False
        elif isinstance(node, intrinsic.Class):
            self.global_effect = False
        elif isinstance(node, intrinsic.UnboundValueType):
            self.global_effect = True
        else:
            raise NotImplementedError