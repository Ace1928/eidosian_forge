from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudidentityGroupsMembershipsLookupRequest(_messages.Message):
    """A CloudidentityGroupsMembershipsLookupRequest object.

  Fields:
    memberKey_id: The ID of the entity. For Google-managed entities, the `id`
      should be the email address of an existing group or user. Email
      addresses need to adhere to [name guidelines for users and
      groups](https://support.google.com/a/answer/9193374). For external-
      identity-mapped entities, the `id` must be a string conforming to the
      Identity Source's requirements. Must be unique within a `namespace`.
    memberKey_namespace: The namespace in which the entity exists. If not
      specified, the `EntityKey` represents a Google-managed entity such as a
      Google user or a Google Group. If specified, the `EntityKey` represents
      an external-identity-mapped group. The namespace must correspond to an
      identity source created in Admin Console and must be in the form of
      `identitysources/{identity_source}`.
    parent: Required. The parent `Group` resource under which to lookup the
      `Membership` name. Must be of the form `groups/{group}`.
  """
    memberKey_id = _messages.StringField(1)
    memberKey_namespace = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)