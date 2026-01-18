from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.core import resources
def GetEffectiveApiEndpoint(release_track=base.ReleaseTrack.ALPHA):
    api_version = _API_VERSION_FOR_TRACK.get(release_track)
    return apis.GetEffectiveApiEndpoint(_API_NAME, api_version)