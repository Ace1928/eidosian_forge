from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.scheduler import jobs
from googlecloudsdk.api_lib.scheduler import locations
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
def GetApiAdapter(release_track, legacy_cron=False):
    if release_track == base.ReleaseTrack.ALPHA:
        return AlphaApiAdapter(legacy_cron=legacy_cron)
    elif release_track == base.ReleaseTrack.BETA:
        return BetaApiAdapter(legacy_cron=legacy_cron)
    elif release_track == base.ReleaseTrack.GA:
        return GaApiAdapter(legacy_cron=legacy_cron)
    else:
        raise UnsupportedReleaseTrackError(release_track)