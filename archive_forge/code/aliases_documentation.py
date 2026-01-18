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

        After an if statement, the values from both branches are merged,
        potentially creating more aliasing:

        >>> from pythran import passmanager
        >>> pm = passmanager.PassManager('demo')
        >>> fun = """
        ... def foo(a, b):
        ...     if a: c=a
        ...     else: c=b
        ...     return {c}"""
        >>> module = ast.parse(fun)
        >>> result = pm.gather(Aliases, module)
        >>> Aliases.dump(result, filter=ast.Set)
        {c} => ['|a|, |b|']
        