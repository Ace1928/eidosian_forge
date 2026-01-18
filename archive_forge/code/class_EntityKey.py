from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EntityKey(_messages.Message):
    """A unique identifier for an entity in the Cloud Identity Groups API. An
  entity can represent either a group with an optional `namespace` or a user
  without a `namespace`. The combination of `id` and `namespace` must be
  unique; however, the same `id` can be used with different `namespace`s.

  Fields:
    id: The ID of the entity. For Google-managed entities, the `id` should be
      the email address of an existing group or user. Email addresses need to
      adhere to [name guidelines for users and
      groups](https://support.google.com/a/answer/9193374). For external-
      identity-mapped entities, the `id` must be a string conforming to the
      Identity Source's requirements. Must be unique within a `namespace`.
    namespace: The namespace in which the entity exists. If not specified, the
      `EntityKey` represents a Google-managed entity such as a Google user or
      a Google Group. If specified, the `EntityKey` represents an external-
      identity-mapped group. The namespace must correspond to an identity
      source created in Admin Console and must be in the form of
      `identitysources/{identity_source}`.
  """
    id = _messages.StringField(1)
    namespace = _messages.StringField(2)