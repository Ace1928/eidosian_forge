from pythran.passmanager import Transformation
from pythran.analyses.ast_matcher import ASTMatcher, AST_any
from pythran.conversion import mangle
from pythran.utils import isnum
import gast as ast
import copy

    Replaces a[:] = b by a call to numpy.copyto.

    This is a slight extension to numpy.copyto as it assumes it also supports
    string and list as first argument.

    >>> import gast as ast
    >>> from pythran import passmanager, backend
    >>> node = ast.parse('a[:] = b')
    >>> pm = passmanager.PassManager("test")
    >>> _, node = pm.apply(CopyTo, node)
    >>> print(pm.dump(backend.Python, node))
    import numpy as __pythran_import_numpy
    __pythran_import_numpy.copyto(a, b)
    >>> node = ast.parse('a[:] = b[:]')
    >>> pm = passmanager.PassManager("test")
    >>> _, node = pm.apply(CopyTo, node)
    >>> print(pm.dump(backend.Python, node))
    import numpy as __pythran_import_numpy
    __pythran_import_numpy.copyto(a, b)
    