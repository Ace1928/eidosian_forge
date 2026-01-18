from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import utils as compute_api
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.resource_policies import flags
from googlecloudsdk.command_lib.compute.resource_policies import util
Create a Compute Engine Snapshot Schedule Resource Policy.

  *{command} creates a Resource Policy which can be attached to disks and
  specifies a schedule for taking disk snapshots and how long these snapshots
  should be retained.
  