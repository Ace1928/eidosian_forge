from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container import api_adapter
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class ContainerAlpha(Container):
    """Deploy and manage clusters of machines for running containers."""

    def Filter(self, context, args):
        """Modify the context that will be given to this group's commands when run.

    Args:
      context: {str:object}, A set of key-value pairs that can be used for
        common initialization among commands.
      args: argparse.Namespace: The same namespace given to the corresponding
        .Run() invocation.

    Returns:
      The refined command context.
    """
        base.DisableUserProjectQuota()
        context['api_adapter'] = api_adapter.NewAPIAdapter('v1alpha1')
        self.EnableSelfSignedJwtForTracks([base.ReleaseTrack.ALPHA])
        return context