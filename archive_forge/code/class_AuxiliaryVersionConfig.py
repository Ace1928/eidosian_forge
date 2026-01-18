from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AuxiliaryVersionConfig(_messages.Message):
    """Configuration information for the auxiliary service versions.

  Messages:
    ConfigOverridesValue: A mapping of Hive metastore configuration key-value
      pairs to apply to the auxiliary Hive metastore (configured in hive-
      site.xml) in addition to the primary version's overrides. If keys are
      present in both the auxiliary version's overrides and the primary
      version's overrides, the value from the auxiliary version's overrides
      takes precedence.

  Fields:
    configOverrides: A mapping of Hive metastore configuration key-value pairs
      to apply to the auxiliary Hive metastore (configured in hive-site.xml)
      in addition to the primary version's overrides. If keys are present in
      both the auxiliary version's overrides and the primary version's
      overrides, the value from the auxiliary version's overrides takes
      precedence.
    networkConfig: Output only. The network configuration contains the
      endpoint URI(s) of the auxiliary Hive metastore service.
    version: The Hive metastore version of the auxiliary service. It must be
      less than the primary Hive metastore service's version.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ConfigOverridesValue(_messages.Message):
        """A mapping of Hive metastore configuration key-value pairs to apply to
    the auxiliary Hive metastore (configured in hive-site.xml) in addition to
    the primary version's overrides. If keys are present in both the auxiliary
    version's overrides and the primary version's overrides, the value from
    the auxiliary version's overrides takes precedence.

    Messages:
      AdditionalProperty: An additional property for a ConfigOverridesValue
        object.

    Fields:
      additionalProperties: Additional properties of type ConfigOverridesValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ConfigOverridesValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    configOverrides = _messages.MessageField('ConfigOverridesValue', 1)
    networkConfig = _messages.MessageField('NetworkConfig', 2)
    version = _messages.StringField(3)