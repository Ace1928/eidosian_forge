from gast import AST, iter_fields, NodeVisitor, Dict, Set
from itertools import permutations
from math import isnan
class ASTMatcher(NodeVisitor):
    """
    Visitor to gather node matching with a given pattern.

    Examples
    --------
    >>> import gast as ast
    >>> code = "[(i, j) for i in range(a) for j in range(b)]"
    >>> pattern = ast.Call(func=ast.Name('range', ctx=ast.Load(),
    ...                                  annotation=None,
    ...                                  type_comment=None),
    ...                    args=AST_any(), keywords=[])
    >>> len(ASTMatcher(pattern).search(ast.parse(code)))
    2
    >>> code = "[(i, j) for i in range(a) for j in range(b)]"
    >>> pattern = ast.Call(func=ast.Name(id=AST_or('range', 'range'),
    ...                                  ctx=ast.Load(),
    ...                                  annotation=None,
    ...                                  type_comment=None),
    ...                    args=AST_any(), keywords=[])
    >>> len(ASTMatcher(pattern).search(ast.parse(code)))
    2
    >>> code = "{1:2, 3:4}"
    >>> pattern = ast.Dict(keys=[ast.Constant(3, None), ast.Constant(1, None)],
    ...                    values=[ast.Constant(4, None),
    ...                            ast.Constant(2, None)])
    >>> len(ASTMatcher(pattern).search(ast.parse(code)))
    1
    >>> code = "{1, 2, 3}"
    >>> pattern = ast.Set(elts=[ast.Constant(3, None),
    ...                         ast.Constant(2, None),
    ...                         ast.Constant(1, None)])
    >>> len(ASTMatcher(pattern).search(ast.parse(code)))
    1
    """

    def __init__(self, pattern):
        """ Basic initialiser saving pattern and initialising result set. """
        self.pattern = pattern
        self.result = set()
        super(ASTMatcher, self).__init__()

    def visit(self, node):
        """
        Visitor looking for matching between current node and pattern.

        If it match, save it else look for a match at lower level keep going.
        """
        if Check(node, dict()).visit(self.pattern):
            self.result.add(node)
        else:
            self.generic_visit(node)

    def search(self, node):
        """ Facility to get values of the matcher for a given node. """
        self.visit(node)
        return self.result

    def match(self, node):
        return Check(node, dict()).visit(self.pattern)