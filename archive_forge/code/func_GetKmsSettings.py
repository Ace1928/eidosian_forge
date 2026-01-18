from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.assured import util
from googlecloudsdk.calliope.base import ReleaseTrack
def GetKmsSettings(release_track):
    return KMS_SETTINGS_MAP.get(release_track)