from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InstanceGroupManagerAllInstancesConfig(_messages.Message):
    """A InstanceGroupManagerAllInstancesConfig object.

  Fields:
    properties: Properties to set on all instances in the group. You can add
      or modify properties using the instanceGroupManagers.patch or
      regionInstanceGroupManagers.patch. After setting allInstancesConfig on
      the group, you must update the group's instances to apply the
      configuration. To apply the configuration, set the group's
      updatePolicy.type field to use proactive updates or use the
      applyUpdatesToInstances method.
  """
    properties = _messages.MessageField('InstancePropertiesPatch', 1)