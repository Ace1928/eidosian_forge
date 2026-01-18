from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class JobMetadata(_messages.Message):
    """Metadata available primarily for filtering jobs. Will be included in the
  ListJob response and Job SUMMARY view.

  Messages:
    UserDisplayPropertiesValue: List of display properties to help UI filter
      jobs.

  Fields:
    bigTableDetails: Identification of a Cloud Bigtable source used in the
      Dataflow job.
    bigqueryDetails: Identification of a BigQuery source used in the Dataflow
      job.
    datastoreDetails: Identification of a Datastore source used in the
      Dataflow job.
    fileDetails: Identification of a File source used in the Dataflow job.
    pubsubDetails: Identification of a Pub/Sub source used in the Dataflow
      job.
    sdkVersion: The SDK version used to run the job.
    spannerDetails: Identification of a Spanner source used in the Dataflow
      job.
    userDisplayProperties: List of display properties to help UI filter jobs.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class UserDisplayPropertiesValue(_messages.Message):
        """List of display properties to help UI filter jobs.

    Messages:
      AdditionalProperty: An additional property for a
        UserDisplayPropertiesValue object.

    Fields:
      additionalProperties: Additional properties of type
        UserDisplayPropertiesValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a UserDisplayPropertiesValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    bigTableDetails = _messages.MessageField('BigTableIODetails', 1, repeated=True)
    bigqueryDetails = _messages.MessageField('BigQueryIODetails', 2, repeated=True)
    datastoreDetails = _messages.MessageField('DatastoreIODetails', 3, repeated=True)
    fileDetails = _messages.MessageField('FileIODetails', 4, repeated=True)
    pubsubDetails = _messages.MessageField('PubSubIODetails', 5, repeated=True)
    sdkVersion = _messages.MessageField('SdkVersion', 6)
    spannerDetails = _messages.MessageField('SpannerIODetails', 7, repeated=True)
    userDisplayProperties = _messages.MessageField('UserDisplayPropertiesValue', 8)