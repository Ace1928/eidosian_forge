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

        >>> import gast as ast
        >>> from pythran import passmanager, backend
        >>> pm = passmanager.PassManager("test")

        >>> node = ast.parse("def foo(a): a[1:][3]")
        >>> _, node = pm.apply(PartialConstantFolding, node)
        >>> _, node = pm.apply(ConstantFolding, node)
        >>> print(pm.dump(backend.Python, node))
        def foo(a):
            a[4]

        >>> node = ast.parse("def foo(a): a[::2][3]")
        >>> _, node = pm.apply(PartialConstantFolding, node)
        >>> _, node = pm.apply(ConstantFolding, node)
        >>> print(pm.dump(backend.Python, node))
        def foo(a):
            a[6]

        >>> node = ast.parse("def foo(a): a[-4:][5]")
        >>> _, node = pm.apply(PartialConstantFolding, node)
        >>> _, node = pm.apply(ConstantFolding, node)
        >>> print(pm.dump(backend.Python, node))
        def foo(a):
            a[1]
        