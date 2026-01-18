from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudidentityGroupsMembershipsCheckTransitiveMembershipRequest(_messages.Message):
    """A CloudidentityGroupsMembershipsCheckTransitiveMembershipRequest object.

  Fields:
    parent: [Resource
      name](https://cloud.google.com/apis/design/resource_names) of the group
      to check the transitive membership in. Format: `groups/{group}`, where
      `group` is the unique id assigned to the Group to which the Membership
      belongs to.
    query: Required. A CEL expression that MUST include member specification.
      This is a `required` field. Certain groups are uniquely identified by
      both a 'member_key_id' and a 'member_key_namespace', which requires an
      additional query input: 'member_key_namespace'. Example query:
      `member_key_id == 'member_key_id_value'`
  """
    parent = _messages.StringField(1, required=True)
    query = _messages.StringField(2)