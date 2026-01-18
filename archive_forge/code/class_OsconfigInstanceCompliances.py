from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class OsconfigInstanceCompliances(base.Group):
    """Report compliance states for OS policies applied to a Compute Engine VM."""