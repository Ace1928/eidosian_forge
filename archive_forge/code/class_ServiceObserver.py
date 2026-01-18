from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServiceObserver(_messages.Message):
    """ServiceObserver is a resource for defining parameters for Service
  Observer (AKA Service-Graph).

  Enums:
    StateValueValuesEnum: Optional. State of Service Observer. The state set
      here applies to the entire project. Disabled by default.

  Messages:
    LabelsValue: Optional. Set of label tags associated with the
      ServiceObserver resource.

  Fields:
    createTime: Output only. The timestamp when the resource was created.
    labels: Optional. Set of label tags associated with the ServiceObserver
      resource.
    name: Required. Name of the ServiceObserver resource. It matches pattern
      `projects/*/locations/global/serviceObserver`. Note: this is a
      Singleton, so the name is derived from the name of the project.
    state: Optional. State of Service Observer. The state set here applies to
      the entire project. Disabled by default.
    updateTime: Output only. The timestamp when the resource was updated.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Optional. State of Service Observer. The state set here applies to the
    entire project. Disabled by default.

    Values:
      STATE_UNSPECIFIED: Service Observer wasn't set. Default is disabled.
      DISABLED: Service Observer is disabled.
      ENABLED: Service Observer is enabled.
    """
        STATE_UNSPECIFIED = 0
        DISABLED = 1
        ENABLED = 2

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. Set of label tags associated with the ServiceObserver
    resource.

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
    labels = _messages.MessageField('LabelsValue', 2)
    name = _messages.StringField(3)
    state = _messages.EnumField('StateValueValuesEnum', 4)
    updateTime = _messages.StringField(5)