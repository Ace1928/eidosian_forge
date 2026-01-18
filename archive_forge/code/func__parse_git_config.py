from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.container.fleet import resources
from googlecloudsdk.command_lib.container.fleet.config_management import utils
from googlecloudsdk.command_lib.container.fleet.features import base
from googlecloudsdk.command_lib.container.fleet.policycontroller import constants
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import yaml
def _parse_git_config(spec_source, msg):
    """Load GitConfig with the parsed config_sync yaml.

  Args:
    spec_source: The config_sync dict loaded from the config-management.yaml
      given by user.
    msg: The Hub messages package.

  Returns:
    git_config: The GitConfig configuration being used in MembershipConfigs
  """
    git_config = msg.ConfigManagementGitConfig()
    if 'syncWait' in spec_source:
        git_config.syncWaitSecs = spec_source['syncWait']
    for field in ['policyDir', 'secretType', 'syncBranch', 'syncRepo', 'syncRev', 'httpsProxy', 'gcpServiceAccountEmail']:
        if field in spec_source:
            setattr(git_config, field, spec_source[field])
    return git_config