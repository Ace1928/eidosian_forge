from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import re
from googlecloudsdk.command_lib.scc import errors
from googlecloudsdk.command_lib.scc import util as scc_util
def ValidateAndGetMuteConfigFullResourceName(args, version):
    """Validates muteConfig full resource name."""
    mute_config = args.mute_config
    resource_pattern = re.compile('(organizations|projects|folders)/.*/muteConfigs/[a-z]([a-z0-9-]{0,61}[a-z0-9])?$')
    regionalized_resource_pattern = re.compile('(organizations|projects|folders)/.*/locations/.*/muteConfigs/[a-z]([a-z0-9-]{0,61}[a-z0-9])?$')
    if regionalized_resource_pattern.match(mute_config):
        return mute_config
    if resource_pattern.match(mute_config):
        if version == 'v2':
            mute_config_components = mute_config.split('/')
            return f'{mute_config_components[0]}/{mute_config_components[1]}/locations/{args.location}/{mute_config_components[2]}/{mute_config_components[3]}'
        else:
            return mute_config
    raise errors.InvalidSCCInputError('Mute config must match the full resource name, or `--organization=`, `--folder=` or `--project=` must be provided.')