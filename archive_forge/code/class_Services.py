from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.run import flags
class Services(base.Group):
    """View and manage your Cloud Run services.

  This set of commands can be used to view and manage your existing Cloud Run
  services.

  To create new deployments, use `{parent_command} deploy`.
  """
    detailed_help = {'EXAMPLES': '\n          To list your deployed services, run:\n\n            $ {command} list\n      '}

    @staticmethod
    def Args(parser):
        """Adds --platform and the various related args."""
        flags.AddPlatformAndLocationFlags(parser)

    def Filter(self, context, args):
        """Runs before command.Run and validates platform with passed args."""
        flags.GetAndValidatePlatform(args, self.ReleaseTrack(), flags.Product.RUN)
        return context