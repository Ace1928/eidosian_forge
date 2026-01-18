from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BinaryauthorizationProjectsPlatformsListRequest(_messages.Message):
    """A BinaryauthorizationProjectsPlatformsListRequest object.

  Fields:
    pageSize: Requested page size. The server may return fewer results than
      requested. If unspecified, the server picks an appropriate default.
    pageToken: A token identifying a page of results the server should return.
      Typically, this is the value of ListPlatformsResponse.next_page_token
      returned from the previous call to the `ListPlatforms` method.
    parent: Required. Contains the name of the resource requested, specified
      using the format `projects/*`.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)