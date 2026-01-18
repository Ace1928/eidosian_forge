from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigatewayGateway(_messages.Message):
    """A Gateway is an API-aware HTTP proxy. It performs API-Method and/or API-
  Consumer specific actions based on an API Config such as authentication,
  policy enforcement, and backend selection.

  Enums:
    StateValueValuesEnum: Output only. The current state of the Gateway.

  Messages:
    LabelsValue: Optional. Resource labels to represent user-provided
      metadata. Refer to cloud documentation on labels for more details.
      https://cloud.google.com/compute/docs/labeling-resources

  Fields:
    apiConfig: Required. Resource name of the API Config for this Gateway.
      Format:
      projects/{project}/locations/global/apis/{api}/configs/{apiConfig}
    createTime: Output only. Created time.
    defaultHostname: Output only. The default API Gateway host name of the
      form `{gateway_id}-{hash}.{region_code}.gateway.dev`.
    displayName: Optional. Display name.
    labels: Optional. Resource labels to represent user-provided metadata.
      Refer to cloud documentation on labels for more details.
      https://cloud.google.com/compute/docs/labeling-resources
    name: Output only. Resource name of the Gateway. Format:
      projects/{project}/locations/{location}/gateways/{gateway}
    state: Output only. The current state of the Gateway.
    updateTime: Output only. Updated time.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The current state of the Gateway.

    Values:
      STATE_UNSPECIFIED: Gateway does not have a state yet.
      CREATING: Gateway is being created.
      ACTIVE: Gateway is running and ready for requests.
      FAILED: Gateway creation failed.
      DELETING: Gateway is being deleted.
      UPDATING: Gateway is being updated.
    """
        STATE_UNSPECIFIED = 0
        CREATING = 1
        ACTIVE = 2
        FAILED = 3
        DELETING = 4
        UPDATING = 5

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. Resource labels to represent user-provided metadata. Refer
    to cloud documentation on labels for more details.
    https://cloud.google.com/compute/docs/labeling-resources

    Messages:
      AdditionalProperty: An additional property for a LabelsValue object.

    Fields:
      additionalProperties: Additional properties of type LabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    apiConfig = _messages.StringField(1)
    createTime = _messages.StringField(2)
    defaultHostname = _messages.StringField(3)
    displayName = _messages.StringField(4)
    labels = _messages.MessageField('LabelsValue', 5)
    name = _messages.StringField(6)
    state = _messages.EnumField('StateValueValuesEnum', 7)
    updateTime = _messages.StringField(8)