from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BigquerydatatransferProjectsLocationsDataSourcesListRequest(_messages.Message):
    """A BigquerydatatransferProjectsLocationsDataSourcesListRequest object.

  Fields:
    pageSize: Page size. The default page size is the maximum value of 1000
      results.
    pageToken: Pagination token, which can be used to request a specific page
      of `ListDataSourcesRequest` list results. For multiple-page results,
      `ListDataSourcesResponse` outputs a `next_page` token, which can be used
      as the `page_token` value to request the next page of list results.
    parent: Required. The BigQuery project id for which data sources should be
      returned. Must be in the form: `projects/{project_id}` or
      `projects/{project_id}/locations/{location_id}`
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)