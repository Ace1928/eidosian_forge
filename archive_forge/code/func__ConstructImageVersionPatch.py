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
def _ConstructImageVersionPatch(update_image_version, release_track=base.ReleaseTrack.GA):
    """Constructs an environment patch for environment image version.

  Args:
    update_image_version: string, the target image version.
    release_track: base.ReleaseTrack, the release track of command. Will dictate
      which Composer client library will be used.

  Returns:
    (str, Environment), the field mask and environment to use for update.
  """
    messages = api_util.GetMessagesModule(release_track=release_track)
    software_config = messages.SoftwareConfig(imageVersion=update_image_version)
    config = messages.EnvironmentConfig(softwareConfig=software_config)
    return ('config.software_config.image_version', messages.Environment(config=config))