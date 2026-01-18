from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1GovernanceEvent(_messages.Message):
    """Payload associated with Governance related log events.

  Enums:
    EventTypeValueValuesEnum: The type of the event.

  Fields:
    entity: Entity resource information if the log event is associated with a
      specific entity.
    eventType: The type of the event.
    message: The log message.
  """

    class EventTypeValueValuesEnum(_messages.Enum):
        """The type of the event.

    Values:
      EVENT_TYPE_UNSPECIFIED: An unspecified event type.
      RESOURCE_IAM_POLICY_UPDATE: Resource IAM policy update event.
      BIGQUERY_TABLE_CREATE: BigQuery table create event.
      BIGQUERY_TABLE_UPDATE: BigQuery table update event.
      BIGQUERY_TABLE_DELETE: BigQuery table delete event.
      BIGQUERY_CONNECTION_CREATE: BigQuery connection create event.
      BIGQUERY_CONNECTION_UPDATE: BigQuery connection update event.
      BIGQUERY_CONNECTION_DELETE: BigQuery connection delete event.
      BIGQUERY_TAXONOMY_CREATE: BigQuery taxonomy created.
      BIGQUERY_POLICY_TAG_CREATE: BigQuery policy tag created.
      BIGQUERY_POLICY_TAG_DELETE: BigQuery policy tag deleted.
      BIGQUERY_POLICY_TAG_SET_IAM_POLICY: BigQuery set iam policy for policy
        tag.
      ACCESS_POLICY_UPDATE: Access policy update event.
      GOVERNANCE_RULE_MATCHED_RESOURCES: Number of resources matched with
        particular Query.
      GOVERNANCE_RULE_SEARCH_LIMIT_EXCEEDS: Rule processing exceeds the
        allowed limit.
      GOVERNANCE_RULE_ERRORS: Rule processing errors.
      GOVERNANCE_RULE_PROCESSING: Governance rule processing Event.
    """
        EVENT_TYPE_UNSPECIFIED = 0
        RESOURCE_IAM_POLICY_UPDATE = 1
        BIGQUERY_TABLE_CREATE = 2
        BIGQUERY_TABLE_UPDATE = 3
        BIGQUERY_TABLE_DELETE = 4
        BIGQUERY_CONNECTION_CREATE = 5
        BIGQUERY_CONNECTION_UPDATE = 6
        BIGQUERY_CONNECTION_DELETE = 7
        BIGQUERY_TAXONOMY_CREATE = 8
        BIGQUERY_POLICY_TAG_CREATE = 9
        BIGQUERY_POLICY_TAG_DELETE = 10
        BIGQUERY_POLICY_TAG_SET_IAM_POLICY = 11
        ACCESS_POLICY_UPDATE = 12
        GOVERNANCE_RULE_MATCHED_RESOURCES = 13
        GOVERNANCE_RULE_SEARCH_LIMIT_EXCEEDS = 14
        GOVERNANCE_RULE_ERRORS = 15
        GOVERNANCE_RULE_PROCESSING = 16
    entity = _messages.MessageField('GoogleCloudDataplexV1GovernanceEventEntity', 1)
    eventType = _messages.EnumField('EventTypeValueValuesEnum', 2)
    message = _messages.StringField(3)