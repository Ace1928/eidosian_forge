from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from googlecloudsdk.api_lib.edge_cloud.container import util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.edge_cloud.container import admin_users
from googlecloudsdk.command_lib.edge_cloud.container import fleet
from googlecloudsdk.command_lib.edge_cloud.container import resource_args
from googlecloudsdk.command_lib.run import flags
from googlecloudsdk.core import resources
def IsOfflineCredential(args):
    """Identify if the command is requesting an offline credential for LCP cluster.

  Args:
    args: command line arguments.

  Returns:
    Boolean, indication of requesting offline credential.
  """
    if flags.FlagIsExplicitlySet(args, 'offline_credential'):
        return True
    return False