from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatapipelinesV1DataflowJobDetails(_messages.Message):
    """Pipeline job details specific to the Dataflow API. This is encapsulated
  here to allow for more executors to store their specific details separately.

  Messages:
    ResourceInfoValue: Cached version of all the metrics of interest for the
      job. This value gets stored here when the job is terminated. As long as
      the job is running, this field is populated from the Dataflow API.

  Fields:
    currentWorkers: Output only. The current number of workers used to run the
      jobs. Only set to a value if the job is still running.
    resourceInfo: Cached version of all the metrics of interest for the job.
      This value gets stored here when the job is terminated. As long as the
      job is running, this field is populated from the Dataflow API.
    sdkVersion: Output only. The SDK version used to run the job.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ResourceInfoValue(_messages.Message):
        """Cached version of all the metrics of interest for the job. This value
    gets stored here when the job is terminated. As long as the job is
    running, this field is populated from the Dataflow API.

    Messages:
      AdditionalProperty: An additional property for a ResourceInfoValue
        object.

    Fields:
      additionalProperties: Additional properties of type ResourceInfoValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ResourceInfoValue object.

      Fields:
        key: Name of the additional property.
        value: A number attribute.
      """
            key = _messages.StringField(1)
            value = _messages.FloatField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    currentWorkers = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    resourceInfo = _messages.MessageField('ResourceInfoValue', 2)
    sdkVersion = _messages.MessageField('GoogleCloudDatapipelinesV1SdkVersion', 3)