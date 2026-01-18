from . import c_ast
def _generate_decl(self, n):
    """ Generation from a Decl node.
        """
    s = ''
    if n.funcspec:
        s = ' '.join(n.funcspec) + ' '
    if n.storage:
        s += ' '.join(n.storage) + ' '
    if n.align:
        s += self.visit(n.align[0]) + ' '
    s += self._generate_type(n.type)
    return s