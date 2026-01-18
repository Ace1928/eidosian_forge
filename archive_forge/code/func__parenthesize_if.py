from . import c_ast
def _parenthesize_if(self, n, condition):
    """ Visits 'n' and returns its string representation, parenthesized
            if the condition function applied to the node returns True.
        """
    s = self._visit_expr(n)
    if condition(n):
        return '(' + s + ')'
    else:
        return s