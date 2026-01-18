from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.netapp import constants
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
def GetSnapshotResourceSpec(source_snapshot_op=False, positional=True):
    """Gets the Resource Spec for Snapshot.

  Args:
    source_snapshot_op: Boolean on whether operation uses snapshot as source or
      not.
    positional: Boolean on whether resource is positional arg ornot

  Returns:
    The Resource Spec for Snapshot
  """
    location_attribute_config = GetLocationAttributeConfig()
    volume_attribute_config = GetVolumeAttributeConfig(positional=False)
    if source_snapshot_op:
        volume_attribute_config.fallthroughs = []
    if not positional:
        location_attribute_config.fallthroughs = [deps.PropertyFallthrough(properties.VALUES.netapp.location)]
    return concepts.ResourceSpec(constants.SNAPSHOTS_COLLECTION, resource_name='snapshot', projectsId=concepts.DEFAULT_PROJECT_ATTRIBUTE_CONFIG, locationsId=location_attribute_config, volumesId=volume_attribute_config, snapshotsId=GetSnapshotAttributeConfig(positional=positional))