from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class Runtimes(base.Group):
    """List runtimes available to Google App Engine."""
    category = base.APP_ENGINE_CATEGORY