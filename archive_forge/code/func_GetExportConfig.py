from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import sys
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.pubsub import util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
import six
from six.moves.urllib.parse import urlparse
def GetExportConfig(args, psl, project, location, requires_seek):
    """Returns an ExportConfig from arguments."""
    if args.export_pubsub_topic is None:
        return None
    desired_state = GetDesiredExportState(args, psl)
    if requires_seek:
        desired_state = psl.ExportConfig.DesiredStateValueValuesEnum.PAUSED
    export_config = psl.ExportConfig(desiredState=desired_state)
    SetExportConfigResources(args, psl, project, location, export_config)
    return export_config