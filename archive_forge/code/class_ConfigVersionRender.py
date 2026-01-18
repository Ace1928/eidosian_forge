from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConfigVersionRender(_messages.Message):
    """Message describing ConfigVersionRender object

  Messages:
    LabelsValue: Optional. Labels as key value pairs Labels are not supported
      for ConfigVersionRenders. They are only supported at the Configs level.

  Fields:
    configVersion: Output only. Resource identifier to the corresponding
      ConfigVersion resource associated with the ConfigVersionRender.
    createTime: Output only. [Output only] Create time stamp
    disabled: Optional. Disabled boolean to determine if a ConfigVersionRender
      acts as a deleted (but recoverable) resource. Default value is False.
    labels: Optional. Labels as key value pairs Labels are not supported for
      ConfigVersionRenders. They are only supported at the Configs level.
    name: Output only. Identifier. [Output only] The resource name of the
      ConfigVersionRender in the format
      `projects/*/configs/*/versionRenders/*`.
    payload: Required. Immutable. Payload content of a ConfigVersion resource.
      If the parent Config has a RAW ConfigType the payload data must point to
      a RawPayload & if the parent Config has a TEMPLATED ConfigType the
      payload data must point to a TemplateValuesPayload. This is only
      returned when the Get/(List?) request provides the View value of FULL.
    renderedPayload: Output only. Server generated rendered version of the
      user provided payload data (ConfigVersionPayload) which has all
      references to a SecretManager version resource substitutions. Any
      TemplateInstances referenced by the ConfigVersions are also processed to
      generate the final templated config output.
    updateTime: Output only. [Output only] Update time stamp
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. Labels as key value pairs Labels are not supported for
    ConfigVersionRenders. They are only supported at the Configs level.

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
    configVersion = _messages.StringField(1)
    createTime = _messages.StringField(2)
    disabled = _messages.BooleanField(3)
    labels = _messages.MessageField('LabelsValue', 4)
    name = _messages.StringField(5)
    payload = _messages.MessageField('ConfigVersionPayload', 6)
    renderedPayload = _messages.BytesField(7)
    updateTime = _messages.StringField(8)