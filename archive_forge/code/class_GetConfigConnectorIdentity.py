from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
from googlecloudsdk.api_lib.container import util as container_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.composer import util as composer_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.GA)
class GetConfigConnectorIdentity(base.Command):
    """Fetch default Config Connector identity.

  {command} prints the default Config Connector Google Service Account in
  a specific Anthos Config Controller.
  """
    detailed_help = {'EXAMPLES': "          To print the default Config Connector identity used by your\n          Config Controller 'main' in the location 'us-central1', run:\n\n            $ {command} main --location=us-central1\n      "}

    @staticmethod
    def Args(parser):
        """Register flags for this command.

    Args:
      parser: An argparse.ArgumentParser-like object. It is mocked out in order
        to capture some information, but behaves like an ArgumentParser.
    """
        parser.add_argument('name', help='Name of the Anthos Config Controller.')
        parser.add_argument('--location', required=True, help='The location (region) of the Anthos Config Controller.')

    def Run(self, args):
        """This is what gets called when the user runs this command.

    Args:
      args: an argparse namespace. All the arguments that were provided to this
        command invocation.
    """
        _BaseRun(args)