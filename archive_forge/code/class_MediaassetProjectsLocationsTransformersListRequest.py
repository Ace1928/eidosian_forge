from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MediaassetProjectsLocationsTransformersListRequest(_messages.Message):
    """A MediaassetProjectsLocationsTransformersListRequest object.

  Fields:
    filter: The filter to apply to list results.
    orderBy: Specifies the ordering of results following syntax at
      https://cloud.google.com/apis/design/design_patterns#sorting_order.
    pageSize: The maximum number of items to return. If unspecified, server
      will pick an appropriate default. Server may return fewer items than
      requested. A caller should only rely on response's next_page_token to
      determine if there are more realms left to be queried
    pageToken: The next_page_token value returned from a previous List
      request, if any.
    parent: Required. The parent resource name, in the following form:
      `projects/{project}/locations/{location}`.
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)