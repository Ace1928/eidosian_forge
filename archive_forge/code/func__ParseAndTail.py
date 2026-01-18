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
def _ParseAndTail(self, tree):
    """Parses an andtail term.

    Args:
      tree: The backend expression tree.

    Returns:
      The new backend expression tree.
    """
    if self._lex.IsString('AND'):
        self._CheckParenthesization(self._OP_AND)
        tree = self._backend.ExprAND(tree, self._ParseOrTerm(must=True))
    return tree