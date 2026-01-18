from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
class InstanceGroups(base.Group):
    """Read and manipulate Compute Engine instance groups.

  Read and manipulate Compute Engine instance groups. To accommodate the
  differences between managed and unmanaged instances, some commands (such as
  `delete`) are in the managed or unmanaged subgroups.
  """