from gast import AST, iter_fields, NodeVisitor, Dict, Set
from itertools import permutations
from math import isnan
class DamnTooLongPattern(Exception):
    """ Exception for long dict/set comparison to reduce compile time. """