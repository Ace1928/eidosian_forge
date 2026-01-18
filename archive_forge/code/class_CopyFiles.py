from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import scp_utils
from googlecloudsdk.core import log
class CopyFiles(base.Command):
    """Copy files to and from Google Compute Engine virtual machines via scp."""

    @staticmethod
    def Args(parser):
        """Set up arguments for this command.

    Args:
      parser: An argparse.ArgumentParser.
    """
        scp_utils.BaseScpHelper.Args(parser)

    def Run(self, args):
        """See scp_utils.BaseScpCommand.Run."""
        log.warning(ENCOURAGE_SCP_INFO)
        holder = base_classes.ComputeApiHolder(self.ReleaseTrack())
        scp_helper = scp_utils.BaseScpHelper()
        return scp_helper.RunScp(holder, args, recursive=True, release_track=self.ReleaseTrack())