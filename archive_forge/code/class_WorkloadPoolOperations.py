from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class WorkloadPoolOperations(base.Group):
    """Manage IAM workload identity pool long running operations.

  Commands for managing IAM workload identity pool long running operations.
  """