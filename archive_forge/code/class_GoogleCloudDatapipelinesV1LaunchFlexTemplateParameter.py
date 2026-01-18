from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatapipelinesV1LaunchFlexTemplateParameter(_messages.Message):
    """Launch Flex Template parameter.

  Messages:
    LaunchOptionsValue: Launch options for this Flex Template job. This is a
      common set of options across languages and templates. This should not be
      used to pass job parameters.
    ParametersValue: The parameters for the Flex Template. Example:
      `{"num_workers":"5"}`
    TransformNameMappingsValue: Use this to pass transform name mappings for
      streaming update jobs. Example:
      `{"oldTransformName":"newTransformName",...}`

  Fields:
    containerSpecGcsPath: Cloud Storage path to a file with a JSON-serialized
      ContainerSpec as content.
    environment: The runtime environment for the Flex Template job.
    jobName: Required. The job name to use for the created job. For an update
      job request, the job name should be the same as the existing running
      job.
    launchOptions: Launch options for this Flex Template job. This is a common
      set of options across languages and templates. This should not be used
      to pass job parameters.
    parameters: The parameters for the Flex Template. Example:
      `{"num_workers":"5"}`
    transformNameMappings: Use this to pass transform name mappings for
      streaming update jobs. Example:
      `{"oldTransformName":"newTransformName",...}`
    update: Set this to true if you are sending a request to update a running
      streaming job. When set, the job name should be the same as the running
      job.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LaunchOptionsValue(_messages.Message):
        """Launch options for this Flex Template job. This is a common set of
    options across languages and templates. This should not be used to pass
    job parameters.

    Messages:
      AdditionalProperty: An additional property for a LaunchOptionsValue
        object.

    Fields:
      additionalProperties: Additional properties of type LaunchOptionsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LaunchOptionsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ParametersValue(_messages.Message):
        """The parameters for the Flex Template. Example: `{"num_workers":"5"}`

    Messages:
      AdditionalProperty: An additional property for a ParametersValue object.

    Fields:
      additionalProperties: Additional properties of type ParametersValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ParametersValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class TransformNameMappingsValue(_messages.Message):
        """Use this to pass transform name mappings for streaming update jobs.
    Example: `{"oldTransformName":"newTransformName",...}`

    Messages:
      AdditionalProperty: An additional property for a
        TransformNameMappingsValue object.

    Fields:
      additionalProperties: Additional properties of type
        TransformNameMappingsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a TransformNameMappingsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    containerSpecGcsPath = _messages.StringField(1)
    environment = _messages.MessageField('GoogleCloudDatapipelinesV1FlexTemplateRuntimeEnvironment', 2)
    jobName = _messages.StringField(3)
    launchOptions = _messages.MessageField('LaunchOptionsValue', 4)
    parameters = _messages.MessageField('ParametersValue', 5)
    transformNameMappings = _messages.MessageField('TransformNameMappingsValue', 6)
    update = _messages.BooleanField(7)