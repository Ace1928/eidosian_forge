from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.resource_policies import util as maintenance_util
from googlecloudsdk.core.util import times
import six
def MakeProjectMapFromProjectList(messages, projects):
    additional_properties = []
    for project in projects:
        additional_properties.append(messages.ShareSettings.ProjectMapValue.AdditionalProperty(key=project, value=messages.ShareSettingsProjectConfig(projectId=project)))
    return messages.ShareSettings.ProjectMapValue(additionalProperties=additional_properties)