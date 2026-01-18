from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudidentityGroupsLookupRequest(_messages.Message):
    """A CloudidentityGroupsLookupRequest object.

  Fields:
    groupKey_id: The ID of the entity. For Google-managed entities, the `id`
      should be the email address of an existing group or user. Email
      addresses need to adhere to [name guidelines for users and
      groups](https://support.google.com/a/answer/9193374). For external-
      identity-mapped entities, the `id` must be a string conforming to the
      Identity Source's requirements. Must be unique within a `namespace`.
    groupKey_namespace: The namespace in which the entity exists. If not
      specified, the `EntityKey` represents a Google-managed entity such as a
      Google user or a Google Group. If specified, the `EntityKey` represents
      an external-identity-mapped group. The namespace must correspond to an
      identity source created in Admin Console and must be in the form of
      `identitysources/{identity_source}`.
  """
    groupKey_id = _messages.StringField(1)
    groupKey_namespace = _messages.StringField(2)