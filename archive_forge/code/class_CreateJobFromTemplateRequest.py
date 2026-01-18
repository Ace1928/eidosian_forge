from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CreateJobFromTemplateRequest(_messages.Message):
    """A request to create a Cloud Dataflow job from a template.

  Messages:
    ParametersValue: The runtime parameters to pass to the job.

  Fields:
    environment: The runtime environment for the job.
    gcsPath: Required. A Cloud Storage path to the template from which to
      create the job. Must be a valid Cloud Storage URL, beginning with
      `gs://`.
    jobName: Required. The job name to use for the created job.
    location: The [regional endpoint]
      (https://cloud.google.com/dataflow/docs/concepts/regional-endpoints) to
      which to direct the request.
    parameters: The runtime parameters to pass to the job.
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
    environment = _messages.MessageField('RuntimeEnvironment', 1)
    gcsPath = _messages.StringField(2)
    jobName = _messages.StringField(3)
    location = _messages.StringField(4)
    parameters = _messages.MessageField('ParametersValue', 5)