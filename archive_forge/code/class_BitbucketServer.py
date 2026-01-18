from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class BitbucketServer(base.Group):
    """Manage Bitbucket Server configurations for Cloud Build."""
    category = base.CI_CD_CATEGORY