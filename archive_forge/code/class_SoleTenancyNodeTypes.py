from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
class SoleTenancyNodeTypes(base.Group):
    """Read Compute Engine sole-tenancy node types.

  Node types are the types of dedicated Compute Engine servers that
  are used for nodes in node groups. Node types will differ in
  the amount of vCPU, Memory, and local SSD space.
  """