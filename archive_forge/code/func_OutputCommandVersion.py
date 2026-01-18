from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.core import resources
def OutputCommandVersion(release_track):
    if release_track == base.ReleaseTrack.GA:
        return ''
    elif release_track == base.ReleaseTrack.BETA:
        return ' beta'
    else:
        return ' alpha'