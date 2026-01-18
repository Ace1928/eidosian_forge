import functools
import gast
def _is_ellipsis_gast_3(node):
    return isinstance(node, gast.Constant) and node.value == Ellipsis