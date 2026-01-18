import re
import copy
import inspect
import ast
import textwrap
class NameVisitor(ast.NodeVisitor):
    """
    NodeVisitor that builds a set of all of the named identifiers in an AST
    """

    def __init__(self, *args, **kwargs):
        super(NameVisitor, self).__init__(*args, **kwargs)
        self.names = set()

    def visit_Name(self, node):
        self.names.add(node.id)

    def visit_arg(self, node):
        if hasattr(node, 'arg'):
            self.names.add(node.arg)
        elif hasattr(node, 'id'):
            self.names.add(node.id)

    def get_new_names(self, num_names):
        """
        Returns a list of new names that are not already present in the AST.

        New names will have the form _N, for N a non-negative integer. If the
        AST has no existing identifiers of this form, then the returned names
        will start at 0 ('_0', '_1', '_2'). If the AST already has identifiers
        of this form, then the names returned will not include the existing
        identifiers.

        Parameters
        ----------
        num_names: int
            The number of new names to return

        Returns
        -------
        list of str
        """
        prop_re = re.compile('^_(\\d+)$')
        matching_names = [n for n in self.names if prop_re.match(n)]
        if matching_names:
            start_number = max([int(n[1:]) for n in matching_names]) + 1
        else:
            start_number = 0
        return ['_' + str(n) for n in range(start_number, start_number + num_names)]