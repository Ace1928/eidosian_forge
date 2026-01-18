from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ResourceReference(_messages.Message):
    """Defines a proto annotation that describes a string field that refers to
  an API resource.

  Fields:
    childType: The resource type of a child collection that the annotated
      field references. This is useful for annotating the `parent` field that
      doesn't have a fixed resource type. Example: message
      ListLogEntriesRequest { string parent = 1
      [(google.api.resource_reference) = { child_type:
      "logging.googleapis.com/LogEntry" }; }
    type: The resource type that the annotated field references. Example:
      message Subscription { string topic = 2 [(google.api.resource_reference)
      = { type: "pubsub.googleapis.com/Topic" }]; } Occasionally, a field may
      reference an arbitrary resource. In this case, APIs use the special
      value * in their resource reference. Example: message
      GetIamPolicyRequest { string resource = 2
      [(google.api.resource_reference) = { type: "*" }]; }
  """
    childType = _messages.StringField(1)
    type = _messages.StringField(2)