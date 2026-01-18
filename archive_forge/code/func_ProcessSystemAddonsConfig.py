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
def ProcessSystemAddonsConfig(args, req):
    """Processes the cluster.system_addons_config.

  Args:
    args: command line arguments.
    req: API request to be issued
  """
    release_track = args.calliope_command.ReleaseTrack()
    msgs = util.GetMessagesModule(release_track)
    data = args.system_addons_config
    try:
        system_addons_config = messages_util.DictToMessageWithErrorCheck(data[GDCE_SYS_ADDONS_CONFIG], msgs.SystemAddonsConfig)
    except (messages_util.DecodeError, AttributeError, KeyError) as err:
        raise exceptions.InvalidArgumentException('--system-addons-config', "'{}'".format(err.args[0] if err.args else err))
    req.cluster.systemAddonsConfig = system_addons_config