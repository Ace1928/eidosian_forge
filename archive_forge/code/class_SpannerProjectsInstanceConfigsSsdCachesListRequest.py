from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SpannerProjectsInstanceConfigsSsdCachesListRequest(_messages.Message):
    """A SpannerProjectsInstanceConfigsSsdCachesListRequest object.

  Fields:
    pageSize: Optional. Number of SSD caches to be returned in the response.
      If 0 or less, defaults to the server's maximum allowed page size.
    pageToken: Optional. If non-empty, `page_token` should contain a
      next_page_token from a previous ListSsdCachesResponse.
    parent: Required. The name of the instance configuration for which a list
      of SSD caches is requested. Values are of the form
      `projects//instanceConfigs/`.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)