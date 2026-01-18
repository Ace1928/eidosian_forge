from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.run import exceptions
from googlecloudsdk.command_lib.run import flags
from googlecloudsdk.command_lib.run import platforms
from googlecloudsdk.core import properties
@base.Hidden
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class Workers(base.Group):
    """View and manage your Cloud Run workers.

  This set of commands can be used to view and manage your Cloud Run workers.
  """
    detailed_help = {'EXAMPLES': '\n          To list your existing workers, run:\n\n            $ {command} list\n      '}

    @staticmethod
    def Args(parser):
        """Adds --region flag."""
        flags.AddRegionArg(parser)

    def Filter(self, context, args):
        """Runs before command.Run and validates platform with passed args."""
        flags.GetAndValidatePlatform(args, self.ReleaseTrack(), flags.Product.RUN)
        return context