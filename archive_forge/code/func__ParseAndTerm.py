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
def _ParseAndTerm(self, must=False):
    """Parses an andterm term.

    Args:
      must: Raises ExpressionSyntaxError if must is True and there is no
        expression.

    Returns:
      The new backend expression tree.
    """
    if self._lex.IsString('NOT'):
        return self._backend.ExprNOT(self._ParseTerm(must=True))
    return self._ParseTerm(must=must)