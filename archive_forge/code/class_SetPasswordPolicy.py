from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.sql import api_util
from googlecloudsdk.api_lib.sql import operations
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.sql import flags
from googlecloudsdk.command_lib.sql import users
from googlecloudsdk.core import properties
@base.ReleaseTracks(base.ReleaseTrack.GA)
class SetPasswordPolicy(base.UpdateCommand):
    """Replaces a user's password policy in a given instance.

  Replaces a user's password policy in a given instance with a specified
  username and host.
  """
    detailed_help = DETAILED_HELP

    @staticmethod
    def Args(parser):
        """Args is called by calliope to gather arguments for this command.

    Args:
      parser: An argparse parser that you can use it to add arguments that go on
        the command line after this command. Positional arguments are allowed.
    """
        AddBaseArgs(parser)

    def Run(self, args):
        RunBaseSetPasswordCommand(args)