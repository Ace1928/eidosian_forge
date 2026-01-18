from pythran.passmanager import NodeAnalysis
import gast as ast
class LocalNodeDeclarations(NodeAnalysis):
    """
    Gathers all local symbols from a function.

    It should not be use from outside a function, but can be used on a function
    (but in that case, parameters are not taken into account)

    >>> import gast as ast
    >>> from pythran import passmanager, backend
    >>> node = ast.parse('''
    ... def foo(a):
    ...     b = a + 1''')
    >>> pm = passmanager.PassManager("test")
    >>> [name.id for name in pm.gather(LocalNodeDeclarations, node)]
    ['b']
    >>> node = ast.parse('''
    ... for c in range(n):
    ...     b = a + 1''')
    >>> pm = passmanager.PassManager("test")
    >>> sorted([name.id for name in pm.gather(LocalNodeDeclarations, node)])
    ['b', 'c']
    """

    def __init__(self):
        """ Initialize empty set as the result. """
        self.result = set()
        super(LocalNodeDeclarations, self).__init__()

    def visit_Name(self, node):
        """ Any node with Store context is a new declaration. """
        if isinstance(node.ctx, ast.Store):
            self.result.add(node)