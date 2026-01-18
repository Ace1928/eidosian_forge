from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BaremetalsolutionProjectsLocationsSnapshotSchedulePoliciesListRequest(_messages.Message):
    """A BaremetalsolutionProjectsLocationsSnapshotSchedulePoliciesListRequest
  object.

  Fields:
    filter: List filter.
    pageSize: The maximum number of items to return.
    pageToken: The next_page_token value returned from a previous List
      request, if any.
    parent: Required. The parent project containing the Snapshot Schedule
      Policies.
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)