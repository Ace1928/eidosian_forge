from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmmigrationProjectsLocationsSourcesUtilizationReportsCreateRequest(_messages.Message):
    """A VmmigrationProjectsLocationsSourcesUtilizationReportsCreateRequest
  object.

  Fields:
    parent: Required. The Utilization Report's parent.
    requestId: A request ID to identify requests. Specify a unique request ID
      so that if you must retry your request, the server will know to ignore
      the request if it has already been completed. The server will guarantee
      that for at least 60 minutes since the first request. For example,
      consider a situation where you make an initial request and the request
      times out. If you make the request again with the same request ID, the
      server can check if original operation with the same request ID was
      received, and if so, will ignore the second request. This prevents
      clients from accidentally creating duplicate commitments. The request ID
      must be a valid UUID with the exception that zero UUID is not supported
      (00000000-0000-0000-0000-000000000000).
    utilizationReport: A UtilizationReport resource to be passed as the
      request body.
    utilizationReportId: Required. The ID to use for the report, which will
      become the final component of the reports's resource name. This value
      maximum length is 63 characters, and valid characters are /a-z-/. It
      must start with an english letter and must not end with a hyphen.
  """
    parent = _messages.StringField(1, required=True)
    requestId = _messages.StringField(2)
    utilizationReport = _messages.MessageField('UtilizationReport', 3)
    utilizationReportId = _messages.StringField(4)