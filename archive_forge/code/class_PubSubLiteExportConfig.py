from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class PubSubLiteExportConfig(_messages.Message):
    """Configuration for a Pub/Sub Lite export subscription.

  Enums:
    StateValueValuesEnum: Output only. An output-only field that indicates
      whether or not the subscription can receive messages.

  Fields:
    serviceAccountEmail: Optional. The service account to use to publish to
      Pub/Sub Lite. The subscription creator or updater that specifies this
      field must have `iam.serviceAccounts.actAs` permission on the service
      account. If not specified, the Pub/Sub [service
      agent](https://cloud.google.com/iam/docs/service-agents),
      service-{project_number}@gcp-sa-pubsub.iam.gserviceaccount.com, is used.
    state: Output only. An output-only field that indicates whether or not the
      subscription can receive messages.
    topic: Optional. The name of the topic to which to write data, of the form
      projects/{project_id}/locations/{location_id}/topics/{topic_id} Pushes
      occur in the same region as the Pub/Sub Lite topic. If this is different
      from the location the messages were published to, egress fees will be
      incurred.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. An output-only field that indicates whether or not the
    subscription can receive messages.

    Values:
      STATE_UNSPECIFIED: Default value. This value is unused.
      ACTIVE: The subscription can actively send messages
      PERMISSION_DENIED: Cannot write to the destination because of permission
        denied errors.
      NOT_FOUND: Cannot write to the destination because it does not exist.
      IN_TRANSIT_LOCATION_RESTRICTION: Cannot write to the destination because
        enforce_in_transit is set to true and the destination locations are
        not in the allowed regions.
    """
        STATE_UNSPECIFIED = 0
        ACTIVE = 1
        PERMISSION_DENIED = 2
        NOT_FOUND = 3
        IN_TRANSIT_LOCATION_RESTRICTION = 4
    serviceAccountEmail = _messages.StringField(1)
    state = _messages.EnumField('StateValueValuesEnum', 2)
    topic = _messages.StringField(3)