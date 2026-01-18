from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import parser_arguments
@base.Hidden
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class StandaloneNodePools(base.Group):
    """Create and manage node pools in an Anthos standalone cluster on bare metal."""