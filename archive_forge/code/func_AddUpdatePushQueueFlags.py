from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from googlecloudsdk.api_lib import tasks as tasks_api_lib
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.tasks import constants
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddUpdatePushQueueFlags(parser, release_track=base.ReleaseTrack.GA, app_engine_queue=False, http_queue=True):
    """Updates flags related to Push queues."""
    if release_track == base.ReleaseTrack.ALPHA:
        flags = _AlphaPushQueueFlags()
    else:
        flags = _PushQueueFlags(release_track)
        if release_track == base.ReleaseTrack.BETA:
            if not app_engine_queue:
                AddQueueTypeFlag(parser)
    if http_queue:
        flags += _HttpPushQueueFlags() + _AddHttpTargetAuthFlags()
    for flag in flags:
        _AddFlagAndItsClearEquivalent(flag, parser)