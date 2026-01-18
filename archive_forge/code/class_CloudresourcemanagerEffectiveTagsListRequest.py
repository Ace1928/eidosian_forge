from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudresourcemanagerEffectiveTagsListRequest(_messages.Message):
    """A CloudresourcemanagerEffectiveTagsListRequest object.

  Fields:
    pageSize: Optional. The maximum number of effective tags to return in the
      response. The server allows a maximum of 300 effective tags to return in
      a single page. If unspecified, the server will use 100 as the default.
    pageToken: Optional. A pagination token returned from a previous call to
      `ListEffectiveTags` that indicates from where this listing should
      continue.
    parent: Required. The full resource name of a resource for which you want
      to list the effective tags. E.g.
      "//cloudresourcemanager.googleapis.com/projects/123"
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3)