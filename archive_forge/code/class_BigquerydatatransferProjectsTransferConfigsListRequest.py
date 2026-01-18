from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BigquerydatatransferProjectsTransferConfigsListRequest(_messages.Message):
    """A BigquerydatatransferProjectsTransferConfigsListRequest object.

  Fields:
    dataSourceIds: When specified, only configurations of requested data
      sources are returned.
    pageSize: Page size. The default page size is the maximum value of 1000
      results.
    pageToken: Pagination token, which can be used to request a specific page
      of `ListTransfersRequest` list results. For multiple-page results,
      `ListTransfersResponse` outputs a `next_page` token, which can be used
      as the `page_token` value to request the next page of list results.
    parent: Required. The BigQuery project id for which transfer configs
      should be returned: `projects/{project_id}` or
      `projects/{project_id}/locations/{location_id}`
  """
    dataSourceIds = _messages.StringField(1, repeated=True)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)