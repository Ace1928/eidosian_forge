from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NotebooksProjectsLocationsRuntimesReportEventRequest(_messages.Message):
    """A NotebooksProjectsLocationsRuntimesReportEventRequest object.

  Fields:
    name: Required. Format:
      `projects/{project_id}/locations/{location}/runtimes/{runtime_id}`
    reportRuntimeEventRequest: A ReportRuntimeEventRequest resource to be
      passed as the request body.
  """
    name = _messages.StringField(1, required=True)
    reportRuntimeEventRequest = _messages.MessageField('ReportRuntimeEventRequest', 2)