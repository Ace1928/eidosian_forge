from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.resource_policies import util as maintenance_util
from googlecloudsdk.core.util import times
import six
def MakeShareSettingsWithDict(messages, dictionary, setting_configs):
    """Constructs the share settings message object from dictionary form of input."""
    if setting_configs:
        if setting_configs == 'organization':
            return messages.ShareSettings(shareType=messages.ShareSettings.ShareTypeValueValuesEnum.ORGANIZATION)
        if setting_configs == 'local':
            if 'share_with' in dictionary.keys():
                raise exceptions.InvalidArgumentException('--share_with', 'The scope this reservation is to be shared with must not be specified with share setting local.')
            return messages.ShareSettings(shareType=messages.ShareSettings.ShareTypeValueValuesEnum.LOCAL)
        if setting_configs == 'projects':
            if 'share_with' not in dictionary.keys():
                raise exceptions.InvalidArgumentException('--share_with', 'The projects this reservation is to be shared with must be specified.')
            return messages.ShareSettings(shareType=messages.ShareSettings.ShareTypeValueValuesEnum.SPECIFIC_PROJECTS, projectMap=MakeProjectMapFromProjectList(messages, dictionary.get('share_with', None)))
        if setting_configs == 'folders':
            if 'share_with' not in dictionary.keys():
                raise exceptions.InvalidArgumentException('--share_with', 'The folders this reservation is to be shared with must be specified.')
            return messages.ShareSettings(shareType=messages.ShareSettings.ShareTypeValueValuesEnum.DIRECT_PROJECTS_UNDER_SPECIFIC_FOLDERS, folderMap=MakeFolderMapFromFolderList(messages, dictionary.get('share_with', None)))
    else:
        if 'share_with' in dictionary.keys():
            raise exceptions.InvalidArgumentException('--share_setting', 'Please specify share setting if specifying share with.')
        return None