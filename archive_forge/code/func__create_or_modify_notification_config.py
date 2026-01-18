from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.transfer import creds_util
from googlecloudsdk.command_lib.transfer import jobs_flag_util
from googlecloudsdk.command_lib.transfer import name_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import times
def _create_or_modify_notification_config(job, args, messages, is_update=False):
    """Creates or modifies transfer NotificationConfig object based on args."""
    notification_pubsub_topic = getattr(args, 'notification_pubsub_topic', None)
    notification_event_types = getattr(args, 'notification_event_types', None)
    notification_payload_format = getattr(args, 'notification_payload_format', None)
    if not (notification_pubsub_topic or notification_event_types or notification_payload_format):
        return
    if notification_pubsub_topic:
        if not job.notificationConfig:
            job.notificationConfig = messages.NotificationConfig(pubsubTopic=notification_pubsub_topic)
        else:
            job.notificationConfig.pubsubTopic = notification_pubsub_topic
    if (notification_event_types or notification_payload_format) and (not job.notificationConfig):
        raise ValueError('Cannot set notification config without --notification-pubsub-topic.')
    if notification_payload_format:
        payload_format_key = notification_payload_format.upper()
        job.notificationConfig.payloadFormat = getattr(messages.NotificationConfig.PayloadFormatValueValuesEnum, payload_format_key)
    elif not is_update:
        job.notificationConfig.payloadFormat = messages.NotificationConfig.PayloadFormatValueValuesEnum.JSON
    if notification_event_types:
        event_types = []
        for event_type_arg in notification_event_types:
            event_type_key = 'TRANSFER_OPERATION_' + event_type_arg.upper()
            event_type = getattr(messages.NotificationConfig.EventTypesValueListEntryValuesEnum, event_type_key)
            event_types.append(event_type)
        job.notificationConfig.eventTypes = event_types
    elif not is_update:
        job.notificationConfig.eventTypes = [messages.NotificationConfig.EventTypesValueListEntryValuesEnum.TRANSFER_OPERATION_SUCCESS, messages.NotificationConfig.EventTypesValueListEntryValuesEnum.TRANSFER_OPERATION_FAILED, messages.NotificationConfig.EventTypesValueListEntryValuesEnum.TRANSFER_OPERATION_ABORTED]