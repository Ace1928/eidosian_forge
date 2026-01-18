from __future__ import absolute_import, print_function
from .Visitor import CythonTransform
from .StringEncoding import EncodedString
from . import Options
from . import PyrexTypes
from ..CodeWriter import ExpressionWriter
from .Errors import warning
def _fmt_expr(self, node):
    writer = ExpressionWriter()
    result = writer.write(node)
    return result