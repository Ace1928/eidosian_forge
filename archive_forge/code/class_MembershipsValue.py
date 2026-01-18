from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class MembershipsValue(_messages.Message):
    """Indicates whether each principal in the binding includes the principal
    specified in the request, either directly or indirectly. Each key
    identifies a principal in the binding, and each value indicates whether
    the principal in the binding includes the principal in the request. For
    example, suppose that a binding includes the following principals: *
    `user:alice@example.com` * `group:product-eng@example.com` The principal
    in the replayed access tuple is `user:bob@example.com`. This user is a
    principal of the group `group:product-eng@example.com`. For the first
    principal in the binding, the key is `user:alice@example.com`, and the
    `membership` field in the value is set to `MEMBERSHIP_NOT_INCLUDED`. For
    the second principal in the binding, the key is `group:product-
    eng@example.com`, and the `membership` field in the value is set to
    `MEMBERSHIP_INCLUDED`.

    Messages:
      AdditionalProperty: An additional property for a MembershipsValue
        object.

    Fields:
      additionalProperties: Additional properties of type MembershipsValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a MembershipsValue object.

      Fields:
        key: Name of the additional property.
        value: A
          GoogleCloudPolicysimulatorV1betaBindingExplanationAnnotatedMembershi
          p attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('GoogleCloudPolicysimulatorV1betaBindingExplanationAnnotatedMembership', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)