import functools
import gast
def _is_constant_gast_3(node):
    return isinstance(node, gast.Constant)