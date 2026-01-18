from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import iso_duration
from googlecloudsdk.core.util import times
def AudioTrackProcessor(tracks):
    """Verify at most two tracks, convert to [int, int]."""
    if len(tracks) > 2:
        raise AudioTrackError('Can not specify more than two audio tracks.')
    return tracks