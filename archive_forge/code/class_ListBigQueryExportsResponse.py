from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListBigQueryExportsResponse(_messages.Message):
    """Response message for listing BigQuery exports.

  Fields:
    bigQueryExports: The BigQuery exports from the specified parent.
    nextPageToken: A token, which can be sent as `page_token` to retrieve the
      next page. If this field is omitted, there are no subsequent pages.
  """
    bigQueryExports = _messages.MessageField('GoogleCloudSecuritycenterV2BigQueryExport', 1, repeated=True)
    nextPageToken = _messages.StringField(2)