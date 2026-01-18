from gast import AST, iter_fields, NodeVisitor, Dict, Set
from itertools import permutations
from math import isnan
class AST_any(AST):
    """ Class to specify we don't care about a field value in ast. """