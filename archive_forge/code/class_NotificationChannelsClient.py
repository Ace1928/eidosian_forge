from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
class NotificationChannelsClient(object):
    """Client for Notification Channels service in the Cloud Monitoring."""

    def __init__(self, client=None, messages=None):
        self.client = client or GetClientInstance()
        self.messages = messages or GetMessagesModule(client)
        self._service = self.client.projects_notificationChannels

    def Create(self, project_ref, channel):
        """Creates an Monitoring Alert Policy."""
        req = self.messages.MonitoringProjectsNotificationChannelsCreateRequest(name=project_ref.RelativeName(), notificationChannel=channel)
        return self._service.Create(req)

    def Get(self, channel_ref):
        req = self.messages.MonitoringProjectsNotificationChannelsGetRequest(name=channel_ref.RelativeName())
        return self._service.Get(req)

    def Update(self, channel_ref, channel, fields=None):
        req = self.messages.MonitoringProjectsNotificationChannelsPatchRequest(name=channel_ref.RelativeName(), notificationChannel=channel, updateMask=fields)
        return self._service.Patch(req)