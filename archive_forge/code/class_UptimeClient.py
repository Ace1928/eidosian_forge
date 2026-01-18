from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
class UptimeClient(object):
    """Client for the Uptime Check service in the Stackdriver Monitoring API."""

    def __init__(self, client=None, messages=None):
        self.client = client or GetClientInstance()
        self.messages = messages or GetMessagesModule(client)
        self._service = self.client.projects_uptimeCheckConfigs

    def Get(self, uptime_check_ref):
        """Gets a Stackdriver uptime check."""
        request = self.messages.MonitoringProjectsUptimeCheckConfigsGetRequest(name=uptime_check_ref.RelativeName())
        return self._service.Get(request)

    def Create(self, project_ref, uptime_check):
        """Creates a Stackdriver uptime check."""
        req = self.messages.MonitoringProjectsUptimeCheckConfigsCreateRequest(parent=project_ref.RelativeName(), uptimeCheckConfig=uptime_check)
        return self._service.Create(req)

    def Update(self, uptime_check_ref, uptime_check, fields=None):
        """Updates a Stackdriver uptime check.

    If fields is not specified, then the uptime check is replaced entirely. If
    fields are specified, then only the fields are replaced.

    Args:
      uptime_check_ref: resources.Resource, Resource reference to the
        uptime_check to be updated.
      uptime_check: Uptime Check, The uptime_check message to be sent with the
        request.
      fields: str, Comma separated list of field masks.

    Returns:
      Uptime Check, The updated Uptime Check.
    """
        request = self.messages.MonitoringProjectsUptimeCheckConfigsPatchRequest(name=uptime_check_ref.RelativeName(), uptimeCheckConfig=uptime_check, updateMask=fields)
        return self._service.Patch(request)