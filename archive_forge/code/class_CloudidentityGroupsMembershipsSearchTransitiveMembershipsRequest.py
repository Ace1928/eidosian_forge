from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudidentityGroupsMembershipsSearchTransitiveMembershipsRequest(_messages.Message):
    """A CloudidentityGroupsMembershipsSearchTransitiveMembershipsRequest
  object.

  Fields:
    pageSize: The default page size is 200 (max 1000).
    pageToken: The next_page_token value returned from a previous list
      request, if any.
    parent: [Resource
      name](https://cloud.google.com/apis/design/resource_names) of the group
      to search transitive memberships in. Format: `groups/{group}`, where
      `group` is the unique ID assigned to the Group.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)