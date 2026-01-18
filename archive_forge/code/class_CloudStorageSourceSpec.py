from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class CloudStorageSourceSpec(_messages.Message):
    """The desired state of the CloudStorageSource.

  Fields:
    bucket: Bucket to subscribe to.
    ceOverrides: CloudEventOverrides defines overrides to control the output
      format and modifications of the event sent to the sink.
    eventTypes: EventTypes to subscribe to. If unspecified, then subscribe to
      all events.
    objectNamePrefix: ObjectNamePrefix limits the notifications to objects
      with this prefix.
    project: Project is the ID of the Google Cloud Project that the PubSub
      Topic exists in. If omitted, defaults to same as the cluster.
    secret: Secret is the credential to use to create the Scheduler Job. If
      not specified, defaults to: Name: google-cloud-key Key: key.json
    serviceAccountName: ServiceAccountName is the k8s service account which
      binds to a google service account. This google service account has
      required permissions to poll from a Cloud Pub/Sub subscription. If not
      specified, defaults to use secret.
    sink: Sink is a reference to an object that will resolve to a domain name
      or a URI directly to use as the sink.
  """
    bucket = _messages.StringField(1)
    ceOverrides = _messages.MessageField('CloudEventOverrides', 2)
    eventTypes = _messages.StringField(3, repeated=True)
    objectNamePrefix = _messages.StringField(4)
    project = _messages.StringField(5)
    secret = _messages.MessageField('SecretKeySelector', 6)
    serviceAccountName = _messages.StringField(7)
    sink = _messages.MessageField('Destination', 8)