from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigatewayApi(_messages.Message):
    """An API that can be served by one or more Gateways.

  Enums:
    StateValueValuesEnum: Output only. State of the API.

  Messages:
    LabelsValue: Optional. Resource labels to represent user-provided
      metadata. Refer to cloud documentation on labels for more details.
      https://cloud.google.com/compute/docs/labeling-resources

  Fields:
    createTime: Output only. Created time.
    displayName: Optional. Display name.
    labels: Optional. Resource labels to represent user-provided metadata.
      Refer to cloud documentation on labels for more details.
      https://cloud.google.com/compute/docs/labeling-resources
    managedService: Optional. Immutable. The name of a Google Managed Service
      ( https://cloud.google.com/service-
      infrastructure/docs/glossary#managed). If not specified, a new Service
      will automatically be created in the same project as this API.
    name: Output only. Resource name of the API. Format:
      projects/{project}/locations/global/apis/{api}
    state: Output only. State of the API.
    updateTime: Output only. Updated time.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. State of the API.

    Values:
      STATE_UNSPECIFIED: API does not have a state yet.
      CREATING: API is being created.
      ACTIVE: API is active.
      FAILED: API creation failed.
      DELETING: API is being deleted.
      UPDATING: API is being updated.
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
    createTime = _messages.StringField(1)
    displayName = _messages.StringField(2)
    labels = _messages.MessageField('LabelsValue', 3)
    managedService = _messages.StringField(4)
    name = _messages.StringField(5)
    state = _messages.EnumField('StateValueValuesEnum', 6)
    updateTime = _messages.StringField(7)