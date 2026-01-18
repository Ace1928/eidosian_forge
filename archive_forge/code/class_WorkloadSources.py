from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.Hidden
@base.ReleaseTracks(base.ReleaseTrack.GA)
class WorkloadSources(base.Group):
    """Manage IAM workload identity pool managed identity workload sources.

   Workload sources define which workloads can attest an identity within a
   pool. When a Workload source is defined for a managed identity, matching
   workloads may receive that specific identity.
  """