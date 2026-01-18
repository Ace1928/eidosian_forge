from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.assured import util
from googlecloudsdk.calliope.base import ReleaseTrack
def CreateResourceSettings(resource_type, release_track):
    resource_settings_message = GetResourceSettings(release_track)
    return resource_settings_message(resourceType=resource_type)