from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1EnvironmentInfrastructureSpecOsImageRuntime(_messages.Message):
    """Software Runtime Configuration to run Analyze.

  Messages:
    PropertiesValue: Optional. Spark properties to provide configuration for
      use in sessions created for this environment. The properties to set on
      daemon config files. Property keys are specified in prefix:property
      format. The prefix must be "spark".

  Fields:
    imageVersion: Required. Dataplex Image version.
    javaLibraries: Optional. List of Java jars to be included in the runtime
      environment. Valid input includes Cloud Storage URIs to Jar binaries.
      For example, gs://bucket-name/my/path/to/file.jar
    properties: Optional. Spark properties to provide configuration for use in
      sessions created for this environment. The properties to set on daemon
      config files. Property keys are specified in prefix:property format. The
      prefix must be "spark".
    pythonPackages: Optional. A list of python packages to be installed. Valid
      formats include Cloud Storage URI to a PIP installable library. For
      example, gs://bucket-name/my/path/to/lib.tar.gz
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class PropertiesValue(_messages.Message):
        """Optional. Spark properties to provide configuration for use in
    sessions created for this environment. The properties to set on daemon
    config files. Property keys are specified in prefix:property format. The
    prefix must be "spark".

    Messages:
      AdditionalProperty: An additional property for a PropertiesValue object.

    Fields:
      additionalProperties: Additional properties of type PropertiesValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a PropertiesValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    imageVersion = _messages.StringField(1)
    javaLibraries = _messages.StringField(2, repeated=True)
    properties = _messages.MessageField('PropertiesValue', 3)
    pythonPackages = _messages.StringField(4, repeated=True)