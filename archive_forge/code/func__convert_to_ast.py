import ast
import textwrap
import gast
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import ast_util
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import qual_names
def _convert_to_ast(n):
    """Converts from a known data type to AST."""
    if isinstance(n, str):
        return gast.Name(id=n, ctx=None, annotation=None, type_comment=None)
    if isinstance(n, qual_names.QN):
        return n.ast()
    if isinstance(n, list):
        return [_convert_to_ast(e) for e in n]
    if isinstance(n, tuple):
        return tuple((_convert_to_ast(e) for e in n))
    return n