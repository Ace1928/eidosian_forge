from pythran.analyses import ConstantExpressions, ASTMatcher
from pythran.passmanager import Transformation
from pythran.tables import MODULES
from pythran.conversion import to_ast, ConversionError, ToNotEval, mangle
from pythran.analyses.ast_matcher import DamnTooLongPattern
from pythran.syntax import PythranSyntaxError
from pythran.utils import isintegral, isnum
from pythran.config import cfg
import builtins
import gast as ast
from copy import deepcopy
import logging
import sys
class ConstantFolding(Transformation):
    """
    Replace constant expression by their evaluation.

    >>> import gast as ast
    >>> from pythran import passmanager, backend
    >>> node = ast.parse("def foo(): return 1+3")
    >>> pm = passmanager.PassManager("test")
    >>> _, node = pm.apply(ConstantFolding, node)
    >>> print(pm.dump(backend.Python, node))
    def foo():
        return 4
    """

    def __init__(self):
        Transformation.__init__(self, ConstantExpressions)

    def prepare(self, node):
        assert isinstance(node, ast.Module)
        self.env = {'builtins': builtins}
        self.consteval = ConstEval(self.env)
        if sys.implementation.name == 'pypy':
            self.env['__builtins__'] = self.env['builtins']
        for module_name in MODULES:
            alias_module_name = mangle(module_name)
            try:
                if module_name == '__dispatch__':
                    self.env[alias_module_name] = DispatchProxy()
                else:
                    self.env[alias_module_name] = __import__(module_name)
            except ImportError:
                pass
        for stmt in node.body:
            if isinstance(stmt, ast.FunctionDef):
                self.env[stmt.name] = FunctionDefWrapper(self.consteval, stmt)
        super(ConstantFolding, self).prepare(node)

    def run(self, node):
        builtins.pythran = PythranBuiltins()
        try:
            return super(ConstantFolding, self).run(node)
        finally:
            del builtins.pythran

    def skip(self, node):
        return node
    visit_Constant = visit_Name = skip
    visit_Attribute = skip

    def visit_Attribute(self, node):
        if node.attr == 'Inf':
            self.update = True
            node.attr = 'inf'
        return node
    visit_List = visit_Set = Transformation.generic_visit
    visit_Dict = visit_Tuple = Transformation.generic_visit

    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Attribute):
                if node.func.value.attr == 'pythran':
                    return Transformation.generic_visit(self, node)
        return self.generic_visit(node)

    def generic_visit(self, node):
        if isinstance(node, ast.expr) and node in self.constant_expressions:
            try:
                value = self.consteval(node)
                new_node = to_ast(value)
                if not ASTMatcher(node).match(new_node):
                    self.update = True
                    return new_node
            except DamnTooLongPattern as e:
                logger.info(str(e) + ', skipping constant folding.')
            except ConversionError as e:
                print('error in constant folding: ', e)
                raise
            except ToNotEval:
                pass
            except Exception as e:
                if not cfg.getboolean('pythran', 'ignore_fold_error'):
                    msg = 'when folding expression, pythran met the following runtime exception:\n>>> {}'
                    raise PythranSyntaxError(msg.format(e), node)
        return Transformation.generic_visit(self, node)