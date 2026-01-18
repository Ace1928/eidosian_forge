from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudidentityGroupsMembershipsSearchTransitiveGroupsRequest(_messages.Message):
    """A CloudidentityGroupsMembershipsSearchTransitiveGroupsRequest object.

  Fields:
    pageSize: The default page size is 200 (max 1000).
    pageToken: The next_page_token value returned from a previous list
      request, if any.
    parent: [Resource
      name](https://cloud.google.com/apis/design/resource_names) of the group
      to search transitive memberships in. Format: `groups/{group}`, where
      `group` is always '-' as this API will search across all groups for a
      given member.
    query: Required. A CEL expression that MUST include member specification
      AND label(s). This is a `required` field. Users can search on label
      attributes of groups. CONTAINS match ('in') is supported on labels.
      Identity-mapped groups are uniquely identified by both a `member_key_id`
      and a `member_key_namespace`, which requires an additional query input:
      `member_key_namespace`. Example query: `member_key_id ==
      'member_key_id_value' && in labels` Query may optionally contain
      equality operators on the parent of the group restricting the search
      within a particular customer, e.g. `parent ==
      'customers/{customer_id}'`. The `customer_id` must begin with "C" (for
      example, 'C046psxkn'). This filtering is only supported for Admins with
      groups read permissons on the input customer. Example query:
      `member_key_id == 'member_key_id_value' && in labels && parent ==
      'customers/C046psxkn'`
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)
    query = _messages.StringField(4)