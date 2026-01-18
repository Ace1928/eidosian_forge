from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class ConfigOverridesValue(_messages.Message):
    """A mapping of Hive metastore configuration key-value pairs to apply to
    the Hive metastore (configured in hive-site.xml). The mappings override
    system defaults (some keys cannot be overridden). These overrides are also
    applied to auxiliary versions and can be further customized in the
    auxiliary version's AuxiliaryVersionConfig.

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