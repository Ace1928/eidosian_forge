from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AssuredworkloadsOrganizationsLocationsWorkloadsListRequest(_messages.Message):
    """A AssuredworkloadsOrganizationsLocationsWorkloadsListRequest object.

  Fields:
    filter: A custom filter for filtering by properties of a workload. At this
      time, only filtering by labels is supported.
    pageSize: Page size.
    pageToken: Page token returned from previous request. Page token contains
      context from previous request. Page token needs to be passed in the
      second and following requests.
    parent: Required. Parent Resource to list workloads from. Must be of the
      form `organizations/{org_id}/locations/{location}`.
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)