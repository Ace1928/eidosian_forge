from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.resource import resource_expr_rewrite
def RewriteGlobal(self, call):
    """Rewrites global restriction <call>."""
    return [{call.term: None}]