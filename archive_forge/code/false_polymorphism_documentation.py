from pythran.passmanager import Transformation
from pythran.analyses import DefUseChains, UseDefChains, Identifiers
import gast as ast

    Rename variable when possible to avoid false polymorphism.

    >>> import gast as ast
    >>> from pythran import passmanager, backend
    >>> node = ast.parse("def foo(): a = 12; a = 'babar'")
    >>> pm = passmanager.PassManager("test")
    >>> _, node = pm.apply(FalsePolymorphism, node)
    >>> print(pm.dump(backend.Python, node))
    def foo():
        a = 12
        a_ = 'babar'
    