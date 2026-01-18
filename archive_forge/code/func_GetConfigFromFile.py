from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from googlecloudsdk.api_lib.beyondcorp.app import util as api_util
from googlecloudsdk.api_lib.util import messages as messages_util
from googlecloudsdk.command_lib.beyondcorp.app import util as command_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.util import files
def GetConfigFromFile(args, request):
    """Read client connector service configuration from file."""
    path = args.config_from_file
    try:
        content_file = files.ReadFileContents(path)
    except files.Error as e:
        raise exceptions.Error('Specified config file path is invalid:\n{}'.format(e))
    data = json.loads(content_file)
    display_name = data['displayName'] if 'displayName' in data else None
    egress_config = data['egress']['peeredVpc'] if 'egress' in data else None
    return ConstructRequest(data['ingress']['config'], egress_config, display_name, args, request)