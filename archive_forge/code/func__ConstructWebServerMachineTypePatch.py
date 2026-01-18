from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.composer import environments_util as environments_api_util
from googlecloudsdk.api_lib.composer import operations_util as operations_api_util
from googlecloudsdk.api_lib.composer import util as api_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.composer import util as command_util
from googlecloudsdk.core import log
import six
def _ConstructWebServerMachineTypePatch(web_server_machine_type, release_track):
    """Constructs an environment patch for Airflow web server machine type.

  Args:
    web_server_machine_type: str or None, machine type used by the Airflow web
      server.
    release_track: base.ReleaseTrack, the release track of command. It dictates
      which Composer client library is used.

  Returns:
    (str, Environment), the field mask and environment to use for update.
  """
    messages = api_util.GetMessagesModule(release_track=release_track)
    config = messages.EnvironmentConfig(webServerConfig=messages.WebServerConfig(machineType=web_server_machine_type))
    return ('config.web_server_config.machine_type', messages.Environment(config=config))