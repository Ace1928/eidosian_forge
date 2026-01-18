from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InstanceGroupManagerStatusAllInstancesConfig(_messages.Message):
    """A InstanceGroupManagerStatusAllInstancesConfig object.

  Fields:
    currentRevision: [Output Only] Current all-instances configuration
      revision. This value is in RFC3339 text format.
    effective: [Output Only] A bit indicating whether this configuration has
      been applied to all managed instances in the group.
  """
    currentRevision = _messages.StringField(1)
    effective = _messages.BooleanField(2)