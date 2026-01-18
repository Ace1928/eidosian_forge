from gast import AST, iter_fields, NodeVisitor, Dict, Set
from itertools import permutations
from math import isnan
class AST_or(AST):
    """
    Class to specify multiple possibles value for a given field in ast.

    Attributes
    ----------
    args: [ast field value]
        List of possible value for a field of an ast.
    """

    def __init__(self, *args):
        """ Initialiser to keep track of arguments. """
        self.args = args
        super(AST_or, self).__init__()