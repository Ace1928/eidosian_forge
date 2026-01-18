from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.edge_cloud.container import util
from googlecloudsdk.api_lib.util import messages as messages_util
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
def SetSystemAddonsConfig(args, request):
    """Sets the cluster.system_addons_config if specified.

  Args:
    args: command line arguments.
    request: API request to be issued
  """
    if args.IsKnownAndSpecified('system_addons_config'):
        ProcessSystemAddonsConfig(args, request)