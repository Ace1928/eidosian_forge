from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.api_lib.spanner import instances
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.spanner import flags
@base.ReleaseTracks(base.ReleaseTrack.GA)
class Move(base.Command):
    """Move the Cloud Spanner instance to the specified instance config."""
    detailed_help = {'EXAMPLES': textwrap.dedent('        To move the Cloud Spanner instance to the target instance configuration, run:\n          $ {command} my-instance-id --target-config=nam3\n        ')}

    @staticmethod
    def Args(parser):
        """Args is called by calliope to gather arguments for this command.

    For `move` command, we have one positional argument, `instanceId`

    Args:
      parser: An argparse parser that you can use to add arguments that go on
        the command line after this command. Positional arguments are allowed.
    """
        flags.Instance().AddToParser(parser)
        flags.TargetConfig().AddToParser(parser)

    def Run(self, args):
        """This is what gets called when the user runs this command.

    Args:
      args: an argparse namespace. From `Args`, we extract command line
        arguments
    """
        instances.Move(args.instance, args.target_config)