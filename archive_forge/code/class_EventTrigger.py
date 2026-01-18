from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class EventTrigger(_messages.Message):
    """An EventTrigger represents an interest in a subset of events occurring
  in a service.

  Fields:
    eventType: Required. The type of event to observe. For example:
      `google.storage.object.finalize` and
      `google.firebase.analytics.event.log`. Event type consists of three
      parts: 1. namespace: The domain name of the organization in reverse-
      domain notation (e.g. `acme.net` appears as `net.acme`) and any
      organization specific subdivisions. If the organization's top-level
      domain is `com`, the top-level domain is omitted (e.g. `google.com`
      appears as `google`). For example, `google.storage` and
      `google.firebase.analytics`. 2. resource type: The type of resource on
      which event occurs. For example, the Google Cloud Storage API includes
      the types `object` and `bucket`. 3. action: The action that generates
      the event. For example, actions for a Google Cloud Storage Object
      include 'finalize' and 'delete'. These parts are lower case and joined
      by '.'.
    resource: Required. The resource(s) from which to observe events, for
      example, `projects/_/buckets/myBucket/objects/{objectPath=**}`. Can be a
      specific resource or use wildcards to match a set of resources.
      Wildcards can either match a single segment in the resource name, using
      '*', or multiple segments, using '**'. For example,
      `projects/myProject/buckets/*/objects/**` would match all objects in all
      buckets in the 'myProject' project. The contents of wildcards can also
      be captured. This is done by assigning it to a variable name in braces.
      For example,
      `projects/myProject/buckets/{bucket_id=*}/objects/{object_path=**}`.
      Additionally, a single segment capture can omit `=*` and a multiple
      segment capture can specify additional structure. For example, the
      following all match the same buckets, but capture different data:
      `projects/myProject/buckets/*/objects/users/*/data/**` `projects/myProje
      ct/buckets/{bucket_id=*}/objects/users/{user_id}/data/{data_path=**}` `p
      rojects/myProject/buckets/{bucket_id}/objects/{object_path=users/*/data/
      **}` Not all syntactically correct values are accepted by all services.
      For example: 1. The authorization model must support it. Google Cloud
      Functions only allows EventTriggers to be deployed that observe
      resources in the same project as the `CloudFunction`. 2. The resource
      type must match the pattern expected for an `event_type`. For example,
      an `EventTrigger` that has an `event_type` of
      "google.pubsub.topic.publish" should have a resource that matches Google
      Cloud Pub/Sub topics. Additionally, some services may support short
      names when creating an `EventTrigger`. These will always be returned in
      the normalized "long" format. See each *service's* documentation for
      supported formats.
    service: The hostname of the service that should be observed. If no string
      is provided, the default service implementing the API will be used. For
      example, `storage.googleapis.com` is the default for all event types in
      the 'google.storage` namespace.
  """
    eventType = _messages.StringField(1)
    resource = _messages.StringField(2)
    service = _messages.StringField(3)