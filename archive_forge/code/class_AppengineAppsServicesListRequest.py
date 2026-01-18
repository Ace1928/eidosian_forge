from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AppengineAppsServicesListRequest(_messages.Message):
    """A AppengineAppsServicesListRequest object.

  Fields:
    pageSize: Maximum results to return per page.
    pageToken: Continuation token for fetching the next page of results.
    parent: Name of the parent Application resource. Example: apps/myapp.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)