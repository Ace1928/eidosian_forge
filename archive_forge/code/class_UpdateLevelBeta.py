from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.accesscontextmanager import levels as levels_api
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.accesscontextmanager import levels
from googlecloudsdk.command_lib.accesscontextmanager import policies
@base.ReleaseTracks(base.ReleaseTrack.BETA)
class UpdateLevelBeta(UpdateLevelGA):
    _API_VERSION = _API_VERSION_PER_TRACK.get('BETA')
    _FEATURE_MASK = _FEATURE_MASK_PER_TRACK.get('BETA')

    @staticmethod
    def Args(parser):
        UpdateLevelGA.ArgsVersioned(parser, release_track='BETA')