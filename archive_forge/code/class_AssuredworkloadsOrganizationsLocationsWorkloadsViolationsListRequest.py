from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AssuredworkloadsOrganizationsLocationsWorkloadsViolationsListRequest(_messages.Message):
    """A AssuredworkloadsOrganizationsLocationsWorkloadsViolationsListRequest
  object.

  Fields:
    filter: Optional. A custom filter for filtering by the Violations
      properties.
    interval_endTime: The end of the time window.
    interval_startTime: The start of the time window.
    pageSize: Optional. Page size.
    pageToken: Optional. Page token returned from previous request.
    parent: Required. The Workload name. Format
      `organizations/{org_id}/locations/{location}/workloads/{workload}`.
  """
    filter = _messages.StringField(1)
    interval_endTime = _messages.StringField(2)
    interval_startTime = _messages.StringField(3)
    pageSize = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(5)
    parent = _messages.StringField(6, required=True)