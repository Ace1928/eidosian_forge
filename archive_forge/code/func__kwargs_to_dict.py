import gast
from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import templates
from tensorflow.python.autograph.utils import ag_logging
def _kwargs_to_dict(self, node):
    """Ties together all keyword and **kwarg arguments in a single dict."""
    if node.keywords:
        return gast.Call(gast.Name('dict', ctx=gast.Load(), annotation=None, type_comment=None), args=(), keywords=node.keywords)
    else:
        return parser.parse_expression('None')