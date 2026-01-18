from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DisksStopGroupAsyncReplicationResource(_messages.Message):
    """A transient resource used in compute.disks.stopGroupAsyncReplication and
  compute.regionDisks.stopGroupAsyncReplication. It is only used to process
  requests and is not persisted.

  Fields:
    resourcePolicy: The URL of the DiskConsistencyGroupPolicy for the group of
      disks to stop. This may be a full or partial URL, such as: -
      https://www.googleapis.com/compute/v1/projects/project/regions/region
      /resourcePolicies/resourcePolicy -
      projects/project/regions/region/resourcePolicies/resourcePolicy -
      regions/region/resourcePolicies/resourcePolicy
  """
    resourcePolicy = _messages.StringField(1)