from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.resource_manager import tags
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class TagValues(base.Group):
    """Create and manipulate TagValues.

    The Resource Manager Service gives you centralized and programmatic
    control over your organization's Tags. As the tag
    administrator, you will be able to create and configure restrictions across
    the tags in your organization or projects. As the tag user, you will be able
    to attach TagValues to different resources.
  """