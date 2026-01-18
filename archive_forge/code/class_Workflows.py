from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib import lifesciences as lib
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA)
class Workflows(base.Group):
    """Commands for Life Sciences pipelines.

  Command to run pipelines.
  """

    def Filter(self, context, args):
        """Setup the API client within the context for this group's commands.

    Args:
      context: {str:object}, A set of key-value pairs that can be used for
          common initialization among commands.
      args: argparse.Namespace: The same namespace given to the corresponding
          .Run() invocation.

    Returns:
      The updated context.
    """
        context[lib.STORAGE_V1_CLIENT_KEY] = apis.GetClientInstance('storage', 'v1')
        return context