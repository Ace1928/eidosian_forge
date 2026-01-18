from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TransferRun(_messages.Message):
    """Represents a data transfer run.

  Enums:
    StateValueValuesEnum: Data transfer run state. Ignored for input requests.

  Messages:
    ParamsValue: Output only. Parameters specific to each data source. For
      more information see the bq tab in the 'Setting up a data transfer'
      section for each data source. For example the parameters for Cloud
      Storage transfers are listed here: https://cloud.google.com/bigquery-
      transfer/docs/cloud-storage-transfer#bq

  Fields:
    dataSourceId: Output only. Data source id.
    destinationDatasetId: Output only. The BigQuery target dataset id.
    emailPreferences: Output only. Email notifications will be sent according
      to these preferences to the email address of the user who owns the
      transfer config this run was derived from.
    endTime: Output only. Time when transfer run ended. Parameter ignored by
      server for input requests.
    errorStatus: Status of the transfer run.
    name: Identifier. The resource name of the transfer run. Transfer run
      names have the form `projects/{project_id}/locations/{location}/transfer
      Configs/{config_id}/runs/{run_id}`. The name is ignored when creating a
      transfer run.
    notificationPubsubTopic: Output only. Pub/Sub topic where a notification
      will be sent after this transfer run finishes. The format for specifying
      a pubsub topic is: `projects/{project_id}/topics/{topic_id}`
    params: Output only. Parameters specific to each data source. For more
      information see the bq tab in the 'Setting up a data transfer' section
      for each data source. For example the parameters for Cloud Storage
      transfers are listed here: https://cloud.google.com/bigquery-
      transfer/docs/cloud-storage-transfer#bq
    runTime: For batch transfer runs, specifies the date and time of the data
      should be ingested.
    schedule: Output only. Describes the schedule of this transfer run if it
      was created as part of a regular schedule. For batch transfer runs that
      are scheduled manually, this is empty. NOTE: the system might choose to
      delay the schedule depending on the current load, so `schedule_time`
      doesn't always match this.
    scheduleTime: Minimum time after which a transfer run can be started.
    startTime: Output only. Time when transfer run was started. Parameter
      ignored by server for input requests.
    state: Data transfer run state. Ignored for input requests.
    updateTime: Output only. Last time the data transfer run state was
      updated.
    userId: Deprecated. Unique ID of the user on whose behalf transfer is
      done.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Data transfer run state. Ignored for input requests.

    Values:
      TRANSFER_STATE_UNSPECIFIED: State placeholder (0).
      PENDING: Data transfer is scheduled and is waiting to be picked up by
        data transfer backend (2).
      RUNNING: Data transfer is in progress (3).
      SUCCEEDED: Data transfer completed successfully (4).
      FAILED: Data transfer failed (5).
      CANCELLED: Data transfer is cancelled (6).
    """
        TRANSFER_STATE_UNSPECIFIED = 0
        PENDING = 1
        RUNNING = 2
        SUCCEEDED = 3
        FAILED = 4
        CANCELLED = 5

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ParamsValue(_messages.Message):
        """Output only. Parameters specific to each data source. For more
    information see the bq tab in the 'Setting up a data transfer' section for
    each data source. For example the parameters for Cloud Storage transfers
    are listed here: https://cloud.google.com/bigquery-transfer/docs/cloud-
    storage-transfer#bq

    Messages:
      AdditionalProperty: An additional property for a ParamsValue object.

    Fields:
      additionalProperties: Properties of the object.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ParamsValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    dataSourceId = _messages.StringField(1)
    destinationDatasetId = _messages.StringField(2)
    emailPreferences = _messages.MessageField('EmailPreferences', 3)
    endTime = _messages.StringField(4)
    errorStatus = _messages.MessageField('Status', 5)
    name = _messages.StringField(6)
    notificationPubsubTopic = _messages.StringField(7)
    params = _messages.MessageField('ParamsValue', 8)
    runTime = _messages.StringField(9)
    schedule = _messages.StringField(10)
    scheduleTime = _messages.StringField(11)
    startTime = _messages.StringField(12)
    state = _messages.EnumField('StateValueValuesEnum', 13)
    updateTime = _messages.StringField(14)
    userId = _messages.IntegerField(15)