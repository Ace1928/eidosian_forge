from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.resource_policies import util as maintenance_util
from googlecloudsdk.core.util import times
import six
def MakeFolderMapFromFolderList(messages, folders):
    additional_properties = []
    for folder in folders:
        additional_properties.append(messages.ShareSettings.FolderMapValue.AdditionalProperty(key=folder, value=messages.ShareSettingsFolderConfig(folderId=folder)))
    return messages.ShareSettings.FolderMapValue(additionalProperties=additional_properties)