from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class ExpectedNextStateValue(_messages.Message):
    """The proto or JSON formatted expected next state of the resource,
    wrapped in a google.protobuf.Any proto, against which the policy rules are
    evaluated. Services not integrated with custom org policy can omit this
    field. Services integrated with custom org policy must populate this field
    for all requests where the API call changes the state of the resource.
    Custom org policy backend uses these attributes to enforce custom org
    policies. When a proto is wrapped, it is generally the One Platform API
    proto. When a JSON string is wrapped, use `google.protobuf.StringValue`
    for the inner value. For create operations, GCP service is expected to
    pass resource from customer request as is. For update/patch operations,
    GCP service is expected to compute the next state with the patch provided
    by the user. See go/custom-constraints-org-policy-integration-guide for
    additional details.

    Messages:
      AdditionalProperty: An additional property for a ExpectedNextStateValue
        object.

    Fields:
      additionalProperties: Properties of the object. Contains field @type
        with type URL.
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a ExpectedNextStateValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('extra_types.JsonValue', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)