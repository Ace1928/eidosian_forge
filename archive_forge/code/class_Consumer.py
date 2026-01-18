from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.Hidden
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class Consumer(base.Group):
    """A group of subcommands for working with Procurement Consumer resources."""
    category = base.COMMERCE_CATEGORY