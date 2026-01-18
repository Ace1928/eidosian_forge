import gast as ast
from copy import deepcopy
from numpy import floating, integer, complexfloating
from pythran.tables import MODULES, attributes
import pythran.typing as typing
from pythran.syntax import PythranSyntaxError
from pythran.utils import isnum
class HasYield(ast.NodeVisitor):

    def __init__(self):
        super(HasYield, self).__init__()
        self.has_yield = False

    def visit_FunctionDef(self, node):
        pass

    def visit_Yield(self, node):
        self.has_yield = True