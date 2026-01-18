from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.tasks import GetApiAdapter
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.tasks import constants
from googlecloudsdk.command_lib.tasks import flags
from googlecloudsdk.command_lib.tasks import parsers
from googlecloudsdk.core import log
@base.Deprecate(is_removed=False, warning='This command group is deprecated. Use `gcloud alpha tasks queues create` instead')
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class AlphaCreateAppEngine(CreateAppEngine):
    """Create a Cloud Tasks queue.

  The flags available to this command represent the fields of a queue that are
  mutable.
  """
    detailed_help = {'DESCRIPTION': '          {description}\n          ', 'EXAMPLES': '          To create a Cloud Tasks queue:\n\n              $ {command} my-queue\n                --max-attempts=10 --max-retry-duration=5s\n                --max-doublings=4 --min-backoff=1s\n                --max-backoff=10s\n                --max-tasks-dispatched-per-second=100\n                --max-concurrent-tasks=10\n                --routing-override=service:abc\n          '}

    def __init__(self, *args, **kwargs):
        super(AlphaCreateAppEngine, self).__init__(*args, **kwargs)
        self.is_alpha = True

    @staticmethod
    def Args(parser):
        flags.AddQueueResourceArg(parser, 'to create')
        flags.AddLocationFlag(parser)
        flags.AddCreatePushQueueFlags(parser, release_track=base.ReleaseTrack.ALPHA, app_engine_queue=True, http_queue=False)