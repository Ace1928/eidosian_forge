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
def _ConstructWebServerPluginsModePatch(support_web_server_plugins, release_track=base.ReleaseTrack.GA):
    """Constructs an environment patch for web server plugins mode patch.

  Args:
    support_web_server_plugins: bool, defines if plugins are enabled or not.
    release_track: base.ReleaseTrack, the release track of command. Will dictate
      which Composer client library will be used.

  Returns:
    (str, Environment), the field mask and environment to use for update.
  """
    messages = api_util.GetMessagesModule(release_track=release_track)
    software_config = messages.SoftwareConfig()
    if support_web_server_plugins:
        software_config.webServerPluginsMode = messages.SoftwareConfig.WebServerPluginsModeValueValuesEnum.PLUGINS_ENABLED
    else:
        software_config.webServerPluginsMode = messages.SoftwareConfig.WebServerPluginsModeValueValuesEnum.PLUGINS_DISABLED
    config = messages.EnvironmentConfig(softwareConfig=software_config)
    return ('config.software_config.web_server_plugins_mode', messages.Environment(config=config))