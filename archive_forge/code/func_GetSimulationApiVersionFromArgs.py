import json
from googlecloudsdk.api_lib.network_management.simulation import Messages
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.anthos import binary_operations
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.credentials.store import GetFreshAccessToken
def GetSimulationApiVersionFromArgs(args):
    """Return API version based on args.

  Args:
    args: The argparse namespace.

  Returns:
    API version (e.g. v1alpha or v1beta).
  """
    release_track = args.calliope_command.ReleaseTrack()
    if release_track == base.ReleaseTrack.ALPHA:
        return 'v1alpha1'
    raise exceptions.InternalError('Unsupported release track.')