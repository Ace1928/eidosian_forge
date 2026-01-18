from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BaremetalsolutionProjectsLocationsSshKeysListRequest(_messages.Message):
    """A BaremetalsolutionProjectsLocationsSshKeysListRequest object.

  Fields:
    pageSize: The maximum number of items to return.
    pageToken: The next_page_token value returned from a previous List
      request, if any.
    parent: Required. The parent containing the SSH keys. Currently, the only
      valid value for the location is "global".
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)