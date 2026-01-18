from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def GetNotificationChannelResourceSpec():
    return concepts.ResourceSpec('monitoring.projects.notificationChannels', resource_name='Notification Channel', notificationChannelsId=NotificationChannelAttributeConfig(), projectsId=concepts.DEFAULT_PROJECT_ATTRIBUTE_CONFIG)