from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core.resource import resource_expr_rewrite
class Rewriter(resource_expr_rewrite.Backend):
    """Database migration resource filter expression rewriter backend.

  Rewriter is used for all database migration related APIs where filtering is
  done only in the server side.
  """

    def Rewrite(self, expression, defaults=None):
        return (None, expression)