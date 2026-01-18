from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TargetPoolsRemoveInstanceRequest(_messages.Message):
    """A TargetPoolsRemoveInstanceRequest object.

  Fields:
    instances: URLs of the instances to be removed from target pool.
  """
    instances = _messages.MessageField('InstanceReference', 1, repeated=True)