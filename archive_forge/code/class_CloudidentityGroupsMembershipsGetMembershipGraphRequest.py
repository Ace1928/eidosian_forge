from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudidentityGroupsMembershipsGetMembershipGraphRequest(_messages.Message):
    """A CloudidentityGroupsMembershipsGetMembershipGraphRequest object.

  Fields:
    parent: Required. [Resource
      name](https://cloud.google.com/apis/design/resource_names) of the group
      to search transitive memberships in. Format: `groups/{group}`, where
      `group` is the unique ID assigned to the Group to which the Membership
      belongs to. group can be a wildcard collection id "-". When a group is
      specified, the membership graph will be constrained to paths between the
      member (defined in the query) and the parent. If a wildcard collection
      is provided, all membership paths connected to the member will be
      returned.
    query: Required. A CEL expression that MUST include member specification
      AND label(s). Certain groups are uniquely identified by both a
      'member_key_id' and a 'member_key_namespace', which requires an
      additional query input: 'member_key_namespace'. Example query:
      `member_key_id == 'member_key_id_value' && in labels`
  """
    parent = _messages.StringField(1, required=True)
    query = _messages.StringField(2)