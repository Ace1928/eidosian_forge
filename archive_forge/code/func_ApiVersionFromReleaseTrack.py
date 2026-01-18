from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.scheduler import jobs
from googlecloudsdk.api_lib.scheduler import locations
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
def ApiVersionFromReleaseTrack(release_track):
    if release_track == base.ReleaseTrack.ALPHA:
        return ALPHA_API_VERSION
    if release_track == base.ReleaseTrack.BETA:
        return BETA_API_VERSION
    if release_track == base.ReleaseTrack.GA:
        return GA_API_VERSION
    else:
        raise UnsupportedReleaseTrackError(release_track)