from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.resource import resource_expr_rewrite
def RewriteNOT(self, expression):
    """Rewrites <NOT expression>."""
    raise OperatorNotSupportedError('The [{}] operator is not supported.'.format('NOT'))