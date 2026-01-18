from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InstanceGroupManagerStatus(_messages.Message):
    """A InstanceGroupManagerStatus object.

  Fields:
    allInstancesConfig: [Output only] Status of all-instances configuration on
      the group.
    autoscaler: [Output Only] The URL of the Autoscaler that targets this
      instance group manager.
    isStable: [Output Only] A bit indicating whether the managed instance
      group is in a stable state. A stable state means that: none of the
      instances in the managed instance group is currently undergoing any type
      of change (for example, creation, restart, or deletion); no future
      changes are scheduled for instances in the managed instance group; and
      the managed instance group itself is not being modified.
    stateful: [Output Only] Stateful status of the given Instance Group
      Manager.
    versionTarget: [Output Only] A status of consistency of Instances'
      versions with their target version specified by version field on
      Instance Group Manager.
  """
    allInstancesConfig = _messages.MessageField('InstanceGroupManagerStatusAllInstancesConfig', 1)
    autoscaler = _messages.StringField(2)
    isStable = _messages.BooleanField(3)
    stateful = _messages.MessageField('InstanceGroupManagerStatusStateful', 4)
    versionTarget = _messages.MessageField('InstanceGroupManagerStatusVersionTarget', 5)