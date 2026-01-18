from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LaunchFlexTemplateParameter(_messages.Message):
    """Launch FlexTemplate Parameter.

  Messages:
    LaunchOptionsValue: Launch options for this flex template job. This is a
      common set of options across languages and templates. This should not be
      used to pass job parameters.
    ParametersValue: The parameters for FlexTemplate. Ex. {"num_workers":"5"}
    TransformNameMappingsValue: Use this to pass transform_name_mappings for
      streaming update jobs. Ex:{"oldTransformName":"newTransformName",...}'

  Fields:
    containerSpec: Spec about the container image to launch.
    containerSpecGcsPath: Cloud Storage path to a file with json serialized
      ContainerSpec as content.
    environment: The runtime environment for the FlexTemplate job
    jobName: Required. The job name to use for the created job. For update job
      request, job name should be same as the existing running job.
    launchOptions: Launch options for this flex template job. This is a common
      set of options across languages and templates. This should not be used
      to pass job parameters.
    parameters: The parameters for FlexTemplate. Ex. {"num_workers":"5"}
    transformNameMappings: Use this to pass transform_name_mappings for
      streaming update jobs. Ex:{"oldTransformName":"newTransformName",...}'
    update: Set this to true if you are sending a request to update a running
      streaming job. When set, the job name should be the same as the running
      job.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LaunchOptionsValue(_messages.Message):
        """Launch options for this flex template job. This is a common set of
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
        """The parameters for FlexTemplate. Ex. {"num_workers":"5"}

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
        """Use this to pass transform_name_mappings for streaming update jobs.
    Ex:{"oldTransformName":"newTransformName",...}'

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
    containerSpec = _messages.MessageField('ContainerSpec', 1)
    containerSpecGcsPath = _messages.StringField(2)
    environment = _messages.MessageField('FlexTemplateRuntimeEnvironment', 3)
    jobName = _messages.StringField(4)
    launchOptions = _messages.MessageField('LaunchOptionsValue', 5)
    parameters = _messages.MessageField('ParametersValue', 6)
    transformNameMappings = _messages.MessageField('TransformNameMappingsValue', 7)
    update = _messages.BooleanField(8)