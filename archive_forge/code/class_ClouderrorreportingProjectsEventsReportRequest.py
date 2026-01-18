from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ClouderrorreportingProjectsEventsReportRequest(_messages.Message):
    """A ClouderrorreportingProjectsEventsReportRequest object.

  Fields:
    projectName: Required. The resource name of the Google Cloud Platform
      project. Written as `projects/{projectId}`, where `{projectId}` is the
      [Google Cloud Platform project
      ID](https://support.google.com/cloud/answer/6158840). Example: //
      `projects/my-project-123`.
    reportedErrorEvent: A ReportedErrorEvent resource to be passed as the
      request body.
  """
    projectName = _messages.StringField(1, required=True)
    reportedErrorEvent = _messages.MessageField('ReportedErrorEvent', 2)