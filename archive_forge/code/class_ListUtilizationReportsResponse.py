from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListUtilizationReportsResponse(_messages.Message):
    """Response message for 'ListUtilizationReports' request.

  Fields:
    nextPageToken: Output only. A token, which can be sent as `page_token` to
      retrieve the next page. If this field is omitted, there are no
      subsequent pages.
    unreachable: Output only. Locations that could not be reached.
    utilizationReports: Output only. The list of reports.
  """
    nextPageToken = _messages.StringField(1)
    unreachable = _messages.StringField(2, repeated=True)
    utilizationReports = _messages.MessageField('UtilizationReport', 3, repeated=True)