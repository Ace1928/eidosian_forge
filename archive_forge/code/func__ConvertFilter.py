from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.firestore import operations
from googlecloudsdk.api_lib.firestore import rewrite_backend
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.firestore import flags
from googlecloudsdk.core import properties
from googlecloudsdk.core.resource import resource_projection_spec
def _ConvertFilter(self, expression, args):
    """Translates user-provided filter spec into one our backend understands.

    Args:
      expression: a filter spec to translate
      args: the args namespace object
    Returns:
      A tuple of string filter specs. The first is the frontend spec for post
      filtering, the second is a spec that the Firestore Admin API understands.
    """
    operation_rewrite_backend = rewrite_backend.OperationsRewriteBackend()
    display_info = args.GetDisplayInfo()
    defaults = resource_projection_spec.ProjectionSpec(symbols=display_info.transforms, aliases=display_info.aliases)
    return operation_rewrite_backend.Rewrite(expression, defaults=defaults)