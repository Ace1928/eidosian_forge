import sys as _sys
import ast as _ast
from ast import boolop, cmpop, excepthandler, expr, expr_context, operator
from ast import slice, stmt, unaryop, mod, AST
from ast import iter_child_nodes, walk
class pattern(AST):
    pass