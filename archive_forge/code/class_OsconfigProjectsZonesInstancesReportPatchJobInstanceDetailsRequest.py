from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class OsconfigProjectsZonesInstancesReportPatchJobInstanceDetailsRequest(_messages.Message):
    """A OsconfigProjectsZonesInstancesReportPatchJobInstanceDetailsRequest
  object.

  Fields:
    reportPatchJobInstanceDetailsRequest: A
      ReportPatchJobInstanceDetailsRequest resource to be passed as the
      request body.
    resource: The instance reporting its status in the form
      `projects/*/zones/*/instances/*`
  """
    reportPatchJobInstanceDetailsRequest = _messages.MessageField('ReportPatchJobInstanceDetailsRequest', 1)
    resource = _messages.StringField(2, required=True)