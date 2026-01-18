from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudresourcemanagerTagBindingsListRequest(_messages.Message):
    """A CloudresourcemanagerTagBindingsListRequest object.

  Fields:
    pageSize: Optional. The maximum number of TagBindings to return in the
      response. The server allows a maximum of 300 TagBindings to return. If
      unspecified, the server will use 100 as the default.
    pageToken: Optional. A pagination token returned from a previous call to
      `ListTagBindings` that indicates where this listing should continue
      from.
    parent: Required. The full resource name of a resource for which you want
      to list existing TagBindings. E.g.
      "//cloudresourcemanager.googleapis.com/projects/123"
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3)