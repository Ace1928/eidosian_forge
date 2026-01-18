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
def ExprOR(self, left, right):
    """Returns an OR expression node."""
    if left:
        left = left.Rewrite()
    if not left:
        return self.Expr(None)
    if right:
        right = right.Rewrite()
    if not right:
        return self.Expr(None)
    return self.Expr(self.RewriteOR(left, right))