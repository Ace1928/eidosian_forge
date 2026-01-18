from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TargetPoolsAddInstanceRequest(_messages.Message):
    """A TargetPoolsAddInstanceRequest object.

  Fields:
    instances: A full or partial URL to an instance to add to this target
      pool. This can be a full or partial URL. For example, the following are
      valid URLs: - https://www.googleapis.com/compute/v1/projects/project-
      id/zones/zone /instances/instance-name - projects/project-
      id/zones/zone/instances/instance-name - zones/zone/instances/instance-
      name
  """
    instances = _messages.MessageField('InstanceReference', 1, repeated=True)