from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EventarcTransformationTypeValueValuesEnum(_messages.Enum):
    """Optional.

    Values:
      EVENTARC_TRANSFORMATION_CONFIG_UNSPECIFIED: The transformation type is
        unknown (e.g. the query parameter value is invalid). This type is only
        relevant for data plane metrics and should not be used by the CLH.
      EVENTARC_NOT_AN_EVENT: The request does not correspond to an event, so
        it should not be transformed. This type is only relevant for data
        plane metrics and should not be used by the CLH.
      EVENTARC_CE_PUBSUB_BINDING: The transformation in which a CloudEvent is
        extracted from the Pub/Sub message (i.e. the Pub/Sub Protocol Binding,
        see https://github.com/google/knative-gcp/blob/main/docs/spec/pubsub-
        protocol-binding.md). This transformation is very generic and should
        be used for any trigger where EventFlow creates the Pub/Sub messages.
        In practice, this means Audit Log events and events from Ingress
        Platform.
      EVENTARC_CUSTOM_PUBSUB: The transformation in which an arbitrary Pub/Sub
        message is converted into a Pub/Sub event, as specified in go/cloud-
        events-on-google-devx-design.
      EVENTARC_GCS_NOTIFICATION: The transformation in which a Cloud Storage
        Pub/Sub Notification (http://cloud/storage/docs/pubsub-notifications)
        is converted into a CloudEvent, as specified in go/gcs-event-
        conversion. This transformation is specific to the Cloud Storage stop-
        gap integration (go/eventarc-gcs-stopgap-detailed-design).
    """
    EVENTARC_TRANSFORMATION_CONFIG_UNSPECIFIED = 0
    EVENTARC_NOT_AN_EVENT = 1
    EVENTARC_CE_PUBSUB_BINDING = 2
    EVENTARC_CUSTOM_PUBSUB = 3
    EVENTARC_GCS_NOTIFICATION = 4