from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class CloudStorage(_messages.Message):
    """Ingestion settings for Cloud Storage.

  Enums:
    StateValueValuesEnum: Output only. An output-only field that indicates the
      state of the Cloud Storage ingestion source.

  Fields:
    avroFormat: Optional. Data from Cloud Storage will be interpreted in Avro
      format.
    bucket: Optional. Cloud Storage bucket. The bucket name must be without
      any prefix like "gs://". See the [bucket naming requirements]
      (https://cloud.google.com/storage/docs/buckets#naming).
    minimumObjectCreateTime: Optional. Only objects with a larger creation
      timestamp will be ingested.
    pubsubAvroFormat: Optional. It will be assumed data from Cloud Storage was
      written via [Cloud Storage
      subscriptions](https://cloud.google.com/pubsub/docs/cloudstorage).
    state: Output only. An output-only field that indicates the state of the
      Cloud Storage ingestion source.
    textFormat: Optional. Data from Cloud Storage will be interpreted as text.
    uriWildcard: Optional. URI wildcard used to match objects that will be
      ingested. If unset, all objects will be ingested. See the [supported
      patterns](https://cloud.google.com/storage/docs/wildcards).
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. An output-only field that indicates the state of the
    Cloud Storage ingestion source.

    Values:
      STATE_UNSPECIFIED: Default value. This value is unused.
      ACTIVE: Ingestion is active.
      CLOUD_STORAGE_PERMISSION_DENIED: Permission denied encountered while
        calling the Cloud Storage API. This can happen if the Pub/Sub SA has
        not been granted the [appropriate
        permissions](https://cloud.google.com/storage/docs/access-control/iam-
        permissions): - storage.objects.list: to list the objects in a bucket.
        - storage.objects.get: to read the objects in a bucket. -
        storage.buckets.get: to verify the bucket exists.
      PUBLISH_PERMISSION_DENIED: Permission denied encountered while
        publishing to the topic. This can happen if the Pub/Sub SA has not
        been granted the [appropriate publish
        permissions](https://cloud.google.com/pubsub/docs/access-
        control#pubsub.publisher)
      BUCKET_NOT_FOUND: The provided Cloud Storage bucket doesn't exist.
      TOO_MANY_OBJECTS: The Cloud Storage bucket has too many objects,
        ingestion will be paused.
      TOO_MANY_ERRORS: Pub/Sub has encountered a large number of errors when
        parsing the objects and attempting to publish. Ingestion will stop.
    """
        STATE_UNSPECIFIED = 0
        ACTIVE = 1
        CLOUD_STORAGE_PERMISSION_DENIED = 2
        PUBLISH_PERMISSION_DENIED = 3
        BUCKET_NOT_FOUND = 4
        TOO_MANY_OBJECTS = 5
        TOO_MANY_ERRORS = 6
    avroFormat = _messages.MessageField('AvroFormat', 1)
    bucket = _messages.StringField(2)
    minimumObjectCreateTime = _messages.StringField(3)
    pubsubAvroFormat = _messages.MessageField('PubSubAvroFormat', 4)
    state = _messages.EnumField('StateValueValuesEnum', 5)
    textFormat = _messages.MessageField('TextFormat', 6)
    uriWildcard = _messages.StringField(7)