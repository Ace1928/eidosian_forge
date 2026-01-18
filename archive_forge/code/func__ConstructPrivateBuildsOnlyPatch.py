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
def _ConstructPrivateBuildsOnlyPatch(support_private_builds_only, release_track=base.ReleaseTrack.GA):
    """Constructs an environment patch to enable/disable private builds only.

  Args:
    support_private_builds_only: bool or None, defines whether the internet
      access is disabled during builds. Can be specified only in
      Composer 3.
    release_track: base.ReleaseTrack, the release track of command. Will dictate
      which Composer client library will be used.

  Returns:
    (str, Environment), the field mask and environment to use for update.
  """
    messages = api_util.GetMessagesModule(release_track=release_track)
    private_environment_config = messages.PrivateEnvironmentConfig()
    config = messages.EnvironmentConfig(privateEnvironmentConfig=private_environment_config)
    update_mask = 'config.private_environment_config.enable_private_builds_only'
    private_environment_config.enablePrivateBuildsOnly = bool(support_private_builds_only)
    return (update_mask, messages.Environment(config=config))