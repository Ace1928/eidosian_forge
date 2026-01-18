from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NotebooksProjectsLocationsInstancesReportRequest(_messages.Message):
    """A NotebooksProjectsLocationsInstancesReportRequest object.

  Fields:
    name: Required. Format:
      `projects/{project_id}/locations/{location}/instances/{instance_id}`
    reportInstanceInfoRequest: A ReportInstanceInfoRequest resource to be
      passed as the request body.
  """
    name = _messages.StringField(1, required=True)
    reportInstanceInfoRequest = _messages.MessageField('ReportInstanceInfoRequest', 2)