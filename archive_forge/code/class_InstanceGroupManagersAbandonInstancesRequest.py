from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InstanceGroupManagersAbandonInstancesRequest(_messages.Message):
    """A InstanceGroupManagersAbandonInstancesRequest object.

  Fields:
    instances: The URLs of one or more instances to abandon. This can be a
      full URL or a partial URL, such as
      zones/[ZONE]/instances/[INSTANCE_NAME].
  """
    instances = _messages.StringField(1, repeated=True)