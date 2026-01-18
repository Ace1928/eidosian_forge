from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ResourceLimit(_messages.Message):
    """Contains information about amount of some resource in the cluster. For
  memory, value should be in GB.

  Fields:
    maximum: Maximum amount of the resource in the cluster.
    minimum: Minimum amount of the resource in the cluster.
    resourceType: Resource name "cpu", "memory" or gpu-specific string.
  """
    maximum = _messages.IntegerField(1)
    minimum = _messages.IntegerField(2)
    resourceType = _messages.StringField(3)