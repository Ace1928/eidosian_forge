from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SparkApplicationEnvironmentConfig(_messages.Message):
    """Represents the SparkApplicationEnvironmentConfig.

  Messages:
    DefaultPropertiesValue: Optional. A map of default Spark properties to
      apply to workloads in this application environment. These defaults may
      be overridden by per-application properties.

  Fields:
    defaultProperties: Optional. A map of default Spark properties to apply to
      workloads in this application environment. These defaults may be
      overridden by per-application properties.
    defaultVersion: Optional. The default Dataproc version to use for
      applications submitted to this application environment
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class DefaultPropertiesValue(_messages.Message):
        """Optional. A map of default Spark properties to apply to workloads in
    this application environment. These defaults may be overridden by per-
    application properties.

    Messages:
      AdditionalProperty: An additional property for a DefaultPropertiesValue
        object.

    Fields:
      additionalProperties: Additional properties of type
        DefaultPropertiesValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a DefaultPropertiesValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    defaultProperties = _messages.MessageField('DefaultPropertiesValue', 1)
    defaultVersion = _messages.StringField(2)