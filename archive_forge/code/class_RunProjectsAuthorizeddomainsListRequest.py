from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RunProjectsAuthorizeddomainsListRequest(_messages.Message):
    """A RunProjectsAuthorizeddomainsListRequest object.

  Fields:
    pageSize: Maximum results to return per page.
    pageToken: Continuation token for fetching the next page of results.
    parent: Name of the parent Project resource. Example:
      `projects/myproject`.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)