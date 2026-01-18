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
def _ParseAdjTerm(self, must=False):
    """Parses an adjterm term.

    Args:
      must: ExpressionSyntaxError if must is True and there is no expression.

    Raises:
      ExpressionSyntaxError: Term expected in expression.

    Returns:
      The new backend expression tree.
    """
    tree = self._ParseOrTerm()
    if tree:
        tree = self._ParseOrTail(tree)
    elif must:
        raise resource_exceptions.ExpressionSyntaxError('Term expected [{0}].'.format(self._lex.Annotate()))
    return tree