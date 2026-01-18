from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import cli_tree
@base.Hidden
@base.ReleaseTracks(base.ReleaseTrack.GA)
class ResourceMaps(base.Group):
    """Command group for resource map related commands."""