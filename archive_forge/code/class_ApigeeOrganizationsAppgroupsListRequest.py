from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsAppgroupsListRequest(_messages.Message):
    """A ApigeeOrganizationsAppgroupsListRequest object.

  Fields:
    filter: The filter expression to be used to get the list of AppGroups,
      where filtering can be done on status, channelId or channelUri of the
      app group. Examples: filter=status=active", filter=channelId=,
      filter=channelUri=
    pageSize: Count of AppGroups a single page can have in the response. If
      unspecified, at most 1000 AppGroups will be returned. The maximum value
      is 1000; values above 1000 will be coerced to 1000.
    pageToken: The starting index record for listing the AppGroups.
    parent: Required. Name of the Apigee organization. Use the following
      structure in your request: `organizations/{org}`.
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)