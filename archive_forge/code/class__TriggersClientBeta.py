from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.eventarc import common
from googlecloudsdk.api_lib.eventarc.base import EventarcClientBase
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.eventarc import types
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import iso_duration
from googlecloudsdk.core.util import times
class _TriggersClientBeta(_BaseTriggersClient):
    """Client for Triggers service in the Eventarc beta API."""

    def BuildTriggerMessage(self, trigger_ref, event_filters, event_filters_path_pattern, event_data_content_type, service_account, destination_message, transport_topic_ref, channel_ref):
        """Builds a Trigger message with the given data.

    Args:
      trigger_ref: Resource, the Trigger to create.
      event_filters: dict or None, the Trigger's event filters.
      event_filters_path_pattern: dict or None, the Trigger's event filters in
        path pattern format. Ignored in Beta.
      event_data_content_type: str or None, the data content type of the
        Trigger's event. Ignored in Beta.
      service_account: str or None, the Trigger's service account.
      destination_message: Destination message or None, the Trigger's
        destination.
      transport_topic_ref: Resource or None, the user-provided transport topic.
      channel_ref: Resource or None, the channel for 3p events. Ignored in Beta.

    Returns:
      A Trigger message with a destination service.
    """
        criteria_messages = [] if event_filters is None else [self._messages.MatchingCriteria(attribute=key, value=value) for key, value in event_filters.items()]
        transport = None
        if transport_topic_ref:
            transport_topic_name = transport_topic_ref.RelativeName()
            pubsub = self._messages.Pubsub(topic=transport_topic_name)
            transport = self._messages.Transport(pubsub=pubsub)
        return self._messages.Trigger(name=trigger_ref.RelativeName(), matchingCriteria=criteria_messages, serviceAccount=service_account, destination=destination_message, transport=transport)

    def BuildCloudRunDestinationMessage(self, destination_run_service, destination_run_job, destination_run_path, destination_run_region):
        """Builds a Destination message for a destination Cloud Run service.

    Args:
      destination_run_service: str or None, the destination Cloud Run service.
      destination_run_job: this destination is not supported in the beta API,
        but is included as an argument here for method consistency with v1.
      destination_run_path: str or None, the path on the destination Cloud Run
        service.
      destination_run_region: str or None, the destination Cloud Run service's
        region.

    Returns:
      A Destination message for a destination Cloud Run service.
    """
        del destination_run_job
        run_message = self._messages.CloudRunService(service=destination_run_service, path=destination_run_path, region=destination_run_region)
        return self._messages.Destination(cloudRunService=run_message)

    def BuildUpdateMask(self, event_filters, event_data_content_type, service_account, destination_run_service, destination_run_job, destination_run_path, destination_run_region):
        """Builds an update mask for updating a trigger.

    Args:
      event_filters: bool, whether to update the event filters.
      event_data_content_type: bool, whether to update the event data content
        type.
      service_account: bool, whether to update the service account.
      destination_run_service: bool, whether to update the destination service.
      destination_run_job: this destination is not supported in the beta API,
        but is included as an argument here for method consistency with v1.
      destination_run_path: bool, whether to update the destination path.
      destination_run_region: bool, whether to update the destination region.

    Returns:
      The update mask as a string.

    Raises:
      NoFieldsSpecifiedError: No fields are being updated.
    """
        del destination_run_job
        update_mask = []
        if destination_run_path:
            update_mask.append('destination.cloudRunService.path')
        if destination_run_region:
            update_mask.append('destination.cloudRunService.region')
        if destination_run_service:
            update_mask.append('destination.cloudRunService.service')
        if event_filters:
            update_mask.append('matchingCriteria')
        if service_account:
            update_mask.append('serviceAccount')
        if event_data_content_type:
            update_mask.append('eventDataContentType')
        if not update_mask:
            raise NoFieldsSpecifiedError('Must specify at least one field to update.')
        return ','.join(update_mask)

    def GetEventType(self, trigger_message):
        """Gets the Trigger's event type."""
        return types.EventFiltersMessageToType(trigger_message.matchingCriteria)