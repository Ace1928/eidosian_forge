from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def GenerateReportRequest(self, error_message, service, version=None, project=None, request_url=None, user=None):
    """Creates a new error event request.

    Args:
      error_message: str, Crash details including stacktrace
      service: str, Name of service
      version: str, Service version, defaults to None
      project: str, Project to report errors to, defaults to current
      request_url: str, The request url that led to the error
      user: str, The user affected by the error

    Returns:
      The request to send.
    """
    service_context = self.api_messages.ServiceContext(service=service, version=version)
    error_event = self.api_messages.ReportedErrorEvent(serviceContext=service_context, message=error_message)
    if request_url or user:
        error_context = self.api_messages.ErrorContext()
        if request_url:
            error_context.httpRequest = self.api_messages.HttpRequestContext(url=request_url)
        if user:
            error_context.user = user
        error_event.context = error_context
    if project is None:
        project = self._GetGcloudProject()
    project_name = self._MakeProjectName(project)
    return self.api_messages.ClouderrorreportingProjectsEventsReportRequest(projectName=project_name, reportedErrorEvent=error_event)