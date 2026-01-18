from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListReferencesRequest(_messages.Message):
    """The ListResourceMetadataRequest request.

  Fields:
    pageSize: The maximum number of items to return. If unspecified, server
      will pick an appropriate default. Server may return fewer items than
      requested. A caller should only rely on response's next_page_token to
      determine if there are more References left to be queried.
    pageToken: The next_page_token value returned from a previous List
      request, if any.
    parent: Required. The parent resource name (target_resource of this
      reference). For example: `//targetservice.googleapis.com/projects/{my-
      project}/locations/{location}/instances/{my-instance}`.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3)