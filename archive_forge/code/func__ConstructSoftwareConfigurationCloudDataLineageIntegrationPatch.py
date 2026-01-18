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
def _ConstructSoftwareConfigurationCloudDataLineageIntegrationPatch(enabled, release_track):
    """Constructs a patch for updating Cloud Data Lineage integration config.

  Args:
    enabled: bool, whether Cloud Data Lineage integration should be enabled.
    release_track: base.ReleaseTrack, the release track of command. It dictates
      which Composer client library is used.

  Returns:
    (str, Environment), the field mask and environment to use for update.
  """
    messages = api_util.GetMessagesModule(release_track=release_track)
    return ('config.software_config.cloud_data_lineage_integration', messages.Environment(config=messages.EnvironmentConfig(softwareConfig=messages.SoftwareConfig(cloudDataLineageIntegration=messages.CloudDataLineageIntegration(enabled=enabled)))))