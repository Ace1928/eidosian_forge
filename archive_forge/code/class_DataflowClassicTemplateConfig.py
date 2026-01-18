from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataflowClassicTemplateConfig(_messages.Message):
    """Dataflow Job information of Classic Template Type. More info about
  dataflow classic templates can be found here
  https://cloud.google.com/dataflow/docs/guides/templates/creating-templates.

  Messages:
    ParametersValue: The runtime parameters to pass to the job.
    TransformNameMappingsValue: Use this to pass transform_name_mappings for
      streaming update jobs. Ex:{"oldTransformName":"newTransformName",...}'.

  Fields:
    environment: The runtime environment for the job.
    gcsPath: A Cloud Storage path to the template from which to create the
      job. Must be valid Cloud Storage URL, beginning with 'gs://'.
    parameters: The runtime parameters to pass to the job.
    transformNameMappings: Use this to pass transform_name_mappings for
      streaming update jobs. Ex:{"oldTransformName":"newTransformName",...}'.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ParametersValue(_messages.Message):
        """The runtime parameters to pass to the job.

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
    Ex:{"oldTransformName":"newTransformName",...}'.

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
    environment = _messages.MessageField('RuntimeEnvironment', 1)
    gcsPath = _messages.StringField(2)
    parameters = _messages.MessageField('ParametersValue', 3)
    transformNameMappings = _messages.MessageField('TransformNameMappingsValue', 4)