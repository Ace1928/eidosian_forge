from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def GetApiFromTrackAndArgs(track, args):
    if args.IsSpecified('location'):
        return 'v2'
    else:
        return GetApiFromTrack(track)