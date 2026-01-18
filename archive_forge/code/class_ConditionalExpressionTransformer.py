import gast
from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import templates
class ConditionalExpressionTransformer(converter.Base):
    """Converts conditional expressions to functional form."""

    def visit_IfExp(self, node):
        template = '\n        ag__.if_exp(\n            test,\n            lambda: true_expr,\n            lambda: false_expr,\n            expr_repr)\n    '
        expr_repr = parser.unparse(node.test, include_encoding_marker=False).strip()
        return templates.replace_as_expression(template, test=node.test, true_expr=node.body, false_expr=node.orelse, expr_repr=gast.Constant(expr_repr, kind=None))