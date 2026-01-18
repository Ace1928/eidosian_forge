from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.iap import util as iap_api
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exc
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.command_lib.iap import exceptions as iap_exc
from googlecloudsdk.core import properties
def ParseIapDestGroupResource(release_track, args):
    """Parses an IAP TCP DestGroup resource from the input arguments.

  Args:
    release_track: base.ReleaseTrack, release track of command.
    args: an argparse namespace. All the arguments that were provided to this
      command invocation.

  Returns:
    The specified IAP TCP DestGroup resource.
  """
    project = properties.VALUES.core.project.GetOrFail()
    group = getattr(args, 'group_name', None)
    if group is None:
        group = args.dest_group
    return iap_api.IapTunnelDestGroupResource(release_track, project, args.region, group)