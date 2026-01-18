from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.resource_manager import tags
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class TagHolds(base.Group):
    """Create and manipulate TagHolds.

    The Resource Manager Service gives you centralized and programmatic
    control over your organization's Tags. As the tag
    administrator, you will be able to create and configure restrictions across
    the tags in your organization or projects, and you will be able to indicate
    the use of a TagValue as a TagHold.
  """