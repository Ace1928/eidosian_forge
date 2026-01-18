from the server back to the client.
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core.resource import resource_exceptions
from googlecloudsdk.core.resource import resource_filter
from googlecloudsdk.core.resource import resource_lex
from googlecloudsdk.core.resource import resource_projection_spec
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.resource import resource_transform
class _Expr(object):
    """An expression rewrite object that evaluates to the rewritten expression."""

    def __init__(self, expr):
        self.expr = expr

    def Rewrite(self):
        """Returns the server side string rewrite of the filter expression."""
        return self.expr