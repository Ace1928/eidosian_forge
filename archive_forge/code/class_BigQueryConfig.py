from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class BigQueryConfig(_messages.Message):
    """Configuration for a BigQuery subscription.

  Enums:
    StateValueValuesEnum: Output only. An output-only field that indicates
      whether or not the subscription can receive messages.

  Fields:
    dropUnknownFields: Optional. When true and use_topic_schema is true, any
      fields that are a part of the topic schema that are not part of the
      BigQuery table schema are dropped when writing to BigQuery. Otherwise,
      the schemas must be kept in sync and any messages with extra fields are
      not written and remain in the subscription's backlog.
    serviceAccountEmail: Optional. The service account to use to write to
      BigQuery. The subscription creator or updater that specifies this field
      must have `iam.serviceAccounts.actAs` permission on the service account.
      If not specified, the Pub/Sub [service
      agent](https://cloud.google.com/iam/docs/service-agents),
      service-{project_number}@gcp-sa-pubsub.iam.gserviceaccount.com, is used.
    state: Output only. An output-only field that indicates whether or not the
      subscription can receive messages.
    table: Optional. The name of the table to which to write data, of the form
      {projectId}.{datasetId}.{tableId}
    useTableSchema: Optional. When true, use the BigQuery table's schema as
      the columns to write to in BigQuery. `use_table_schema` and
      `use_topic_schema` cannot be enabled at the same time.
    useTopicSchema: Optional. When true, use the topic's schema as the columns
      to write to in BigQuery, if it exists. `use_topic_schema` and
      `use_table_schema` cannot be enabled at the same time.
    writeMetadata: Optional. When true, write the subscription name,
      message_id, publish_time, attributes, and ordering_key to additional
      columns in the table. The subscription name, message_id, and
      publish_time fields are put in their own columns while all other message
      properties (other than data) are written to a JSON object in the
      attributes column.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. An output-only field that indicates whether or not the
    subscription can receive messages.

    Values:
      STATE_UNSPECIFIED: Default value. This value is unused.
      ACTIVE: The subscription can actively send messages to BigQuery
      PERMISSION_DENIED: Cannot write to the BigQuery table because of
        permission denied errors. This can happen if - Pub/Sub SA has not been
        granted the [appropriate BigQuery IAM
        permissions](https://cloud.google.com/pubsub/docs/create-
        subscription#assign_bigquery_service_account) -
        bigquery.googleapis.com API is not enabled for the project
        ([instructions](https://cloud.google.com/service-usage/docs/enable-
        disable))
      NOT_FOUND: Cannot write to the BigQuery table because it does not exist.
      SCHEMA_MISMATCH: Cannot write to the BigQuery table due to a schema
        mismatch.
      IN_TRANSIT_LOCATION_RESTRICTION: Cannot write to the destination because
        enforce_in_transit is set to true and the destination locations are
        not in the allowed regions.
    """
        STATE_UNSPECIFIED = 0
        ACTIVE = 1
        PERMISSION_DENIED = 2
        NOT_FOUND = 3
        SCHEMA_MISMATCH = 4
        IN_TRANSIT_LOCATION_RESTRICTION = 5
    dropUnknownFields = _messages.BooleanField(1)
    serviceAccountEmail = _messages.StringField(2)
    state = _messages.EnumField('StateValueValuesEnum', 3)
    table = _messages.StringField(4)
    useTableSchema = _messages.BooleanField(5)
    useTopicSchema = _messages.BooleanField(6)
    writeMetadata = _messages.BooleanField(7)