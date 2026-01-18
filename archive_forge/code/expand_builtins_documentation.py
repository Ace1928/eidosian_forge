from pythran.analyses import Globals, Locals
from pythran.passmanager import Transformation
from pythran.syntax import PythranSyntaxError
from pythran.tables import MODULES
import builtins
import gast as ast

    Expands all builtins into full paths.

    >>> import gast as ast
    >>> from pythran import passmanager, backend
    >>> node = ast.parse("def foo(): return list()")
    >>> pm = passmanager.PassManager("test")
    >>> _, node = pm.apply(ExpandBuiltins, node)
    >>> print(pm.dump(backend.Python, node))
    def foo():
        return builtins.list()
    