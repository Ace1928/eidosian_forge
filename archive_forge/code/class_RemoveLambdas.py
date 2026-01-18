from pythran.analyses import GlobalDeclarations, ImportedIds
from pythran.analyses import Check
from pythran.analyses import ExtendedDefUseChains
from pythran.passmanager import Transformation
from pythran.tables import MODULES
from pythran.conversion import mangle
import pythran.metadata as metadata
from copy import copy, deepcopy
import gast as ast
class RemoveLambdas(Transformation):
    """
    Turns lambda into top-level functions.

    >>> import gast as ast
    >>> from pythran import passmanager, backend
    >>> node = ast.parse("def foo(y): lambda x:y+x")
    >>> pm = passmanager.PassManager("test")
    >>> _, node = pm.apply(RemoveLambdas, node)
    >>> print(pm.dump(backend.Python, node))
    import functools as __pythran_import_functools
    def foo(y):
        __pythran_import_functools.partial(foo_lambda0, y)
    def foo_lambda0(y, x):
        return (y + x)
    """

    def __init__(self):
        super(RemoveLambdas, self).__init__(GlobalDeclarations)

    def visit_Module(self, node):
        self.lambda_functions = list()
        self.patterns = {}
        self.imports = list()
        self.generic_visit(node)
        node.body = self.imports + node.body + self.lambda_functions
        self.update |= bool(self.imports) or bool(self.lambda_functions)
        return node

    def visit_FunctionDef(self, node):
        lr = _LambdaRemover(self, node.name)
        node.body = [lr.visit(n) for n in node.body]
        return node