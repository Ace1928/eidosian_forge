from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EvaluateUserConsentsRequest(_messages.Message):
    """Evaluate a user's Consents for all matching User data mappings. Note:
  User data mappings are indexed asynchronously, causing slight delays between
  the time mappings are created or updated and when they are included in
  EvaluateUserConsents results.

  Enums:
    ResponseViewValueValuesEnum: Optional. The view for
      EvaluateUserConsentsResponse. If unspecified, defaults to `BASIC` and
      returns `consented` as `TRUE` or `FALSE`.

  Messages:
    RequestAttributesValue: Required. The values of request attributes
      associated with this access request.
    ResourceAttributesValue: Optional. The values of resource attributes
      associated with the resources being requested. If no values are
      specified, then all resources are queried.

  Fields:
    consentList: Optional. Specific Consents to evaluate the access request
      against. These Consents must have the same `user_id` as the User data
      mappings being evalauted, must exist in the current `consent_store`, and
      must have a `state` of either `ACTIVE` or `DRAFT`. A maximum of 100
      Consents can be provided here. If unspecified, all `ACTIVE` unexpired
      Consents in the current `consent_store` will be evaluated.
    pageSize: Optional. Limit on the number of User data mappings to return in
      a single response. If not specified, 100 is used. May not be larger than
      1000.
    pageToken: Optional. Token to retrieve the next page of results, or empty
      to get the first page.
    requestAttributes: Required. The values of request attributes associated
      with this access request.
    resourceAttributes: Optional. The values of resource attributes associated
      with the resources being requested. If no values are specified, then all
      resources are queried.
    responseView: Optional. The view for EvaluateUserConsentsResponse. If
      unspecified, defaults to `BASIC` and returns `consented` as `TRUE` or
      `FALSE`.
    userId: Required. User ID to evaluate consents for.
  """

    class ResponseViewValueValuesEnum(_messages.Enum):
        """Optional. The view for EvaluateUserConsentsResponse. If unspecified,
    defaults to `BASIC` and returns `consented` as `TRUE` or `FALSE`.

    Values:
      RESPONSE_VIEW_UNSPECIFIED: No response view specified. The API will
        default to the BASIC view.
      BASIC: Only the `data_id` and `consented` fields are populated in the
        response.
      FULL: All fields within the response are populated. When set to `FULL`,
        all `ACTIVE` Consents are evaluated even if a matching policy is found
        during evaluation.
    """
        RESPONSE_VIEW_UNSPECIFIED = 0
        BASIC = 1
        FULL = 2

    @encoding.MapUnrecognizedFields('additionalProperties')
    class RequestAttributesValue(_messages.Message):
        """Required. The values of request attributes associated with this access
    request.

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

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ResourceAttributesValue(_messages.Message):
        """Optional. The values of resource attributes associated with the
    resources being requested. If no values are specified, then all resources
    are queried.

    Messages:
      AdditionalProperty: An additional property for a ResourceAttributesValue
        object.

    Fields:
      additionalProperties: Additional properties of type
        ResourceAttributesValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ResourceAttributesValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    consentList = _messages.MessageField('ConsentList', 1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    requestAttributes = _messages.MessageField('RequestAttributesValue', 4)
    resourceAttributes = _messages.MessageField('ResourceAttributesValue', 5)
    responseView = _messages.EnumField('ResponseViewValueValuesEnum', 6)
    userId = _messages.StringField(7)