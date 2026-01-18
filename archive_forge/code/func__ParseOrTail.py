from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.core.resource import resource_exceptions
from googlecloudsdk.core.resource import resource_expr
from googlecloudsdk.core.resource import resource_lex
from googlecloudsdk.core.resource import resource_projection_spec
from googlecloudsdk.core.resource import resource_property
import six
def _ParseOrTail(self, tree):
    """Parses an ortail term.

    Args:
      tree: The backend expression tree.

    Returns:
      The new backend expression tree.
    """
    if self._lex.IsString('OR'):
        self._CheckParenthesization(self._OP_OR)
        tree = self._backend.ExprOR(tree, self._ParseAdjTerm(must=True))
    return tree