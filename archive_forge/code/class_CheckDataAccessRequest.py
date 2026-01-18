from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CheckDataAccessRequest(_messages.Message):
    """Checks if a particular data_id of a User data mapping in the given
  consent store is consented for a given use.

  Enums:
    ResponseViewValueValuesEnum: Optional. The view for
      CheckDataAccessResponse. If unspecified, defaults to `BASIC` and returns
      `consented` as `TRUE` or `FALSE`.

  Messages:
    RequestAttributesValue: The values of request attributes associated with
      this access request.

  Fields:
    consentList: Optional. Specific Consents to evaluate the access request
      against. These Consents must have the same `user_id` as the evaluated
      User data mapping, must exist in the current `consent_store`, and have a
      `state` of either `ACTIVE` or `DRAFT`. A maximum of 100 Consents can be
      provided here. If no selection is specified, the access request is
      evaluated against all `ACTIVE` unexpired Consents with the same
      `user_id` as the evaluated User data mapping.
    dataId: Required. The unique identifier of the resource to check access
      for. This identifier must correspond to a User data mapping in the given
      consent store.
    requestAttributes: The values of request attributes associated with this
      access request.
    responseView: Optional. The view for CheckDataAccessResponse. If
      unspecified, defaults to `BASIC` and returns `consented` as `TRUE` or
      `FALSE`.
  """

    class ResponseViewValueValuesEnum(_messages.Enum):
        """Optional. The view for CheckDataAccessResponse. If unspecified,
    defaults to `BASIC` and returns `consented` as `TRUE` or `FALSE`.

    Values:
      RESPONSE_VIEW_UNSPECIFIED: No response view specified. The API will
        default to the BASIC view.
      BASIC: Only the `consented` field is populated in
        CheckDataAccessResponse.
      FULL: All fields within CheckDataAccessResponse are populated. When set
        to `FULL`, all `ACTIVE` Consents are evaluated even if a matching
        policy is found during evaluation.
    """
        RESPONSE_VIEW_UNSPECIFIED = 0
        BASIC = 1
        FULL = 2

    @encoding.MapUnrecognizedFields('additionalProperties')
    class RequestAttributesValue(_messages.Message):
        """The values of request attributes associated with this access request.

    Messages:
      AdditionalProperty: An additional property for a RequestAttributesValue
        object.

    Fields:
      additionalProperties: Additional properties of type
        RequestAttributesValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a RequestAttributesValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    consentList = _messages.MessageField('ConsentList', 1)
    dataId = _messages.StringField(2)
    requestAttributes = _messages.MessageField('RequestAttributesValue', 3)
    responseView = _messages.EnumField('ResponseViewValueValuesEnum', 4)