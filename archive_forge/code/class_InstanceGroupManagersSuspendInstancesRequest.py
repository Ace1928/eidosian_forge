from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InstanceGroupManagersSuspendInstancesRequest(_messages.Message):
    """A InstanceGroupManagersSuspendInstancesRequest object.

  Fields:
    forceSuspend: If this flag is set to true, the Instance Group Manager will
      proceed to suspend the instances, skipping initialization on them.
    instances: The URLs of one or more instances to suspend. This can be a
      full URL or a partial URL, such as
      zones/[ZONE]/instances/[INSTANCE_NAME].
  """
    forceSuspend = _messages.BooleanField(1)
    instances = _messages.StringField(2, repeated=True)