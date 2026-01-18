from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.tasks import GetApiAdapter
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.tasks import flags
from googlecloudsdk.command_lib.tasks import parsers
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
class Purge(base.Command):
    """Purge a queue by deleting all of its tasks.

  This command purges a queue by deleting all of its tasks. Purge operations can
  take up to one minute to take effect. Tasks might be dispatched before the
  purge takes effect. A purge is irreversible. All tasks created before this
  command is run are permanently deleted.
  """
    detailed_help = {'DESCRIPTION': '          {description}\n          ', 'EXAMPLES': '          To purge a queue:\n\n              $ {command} my-queue\n         '}

    @staticmethod
    def Args(parser):
        flags.AddQueueResourceArg(parser, 'to purge')
        flags.AddLocationFlag(parser)

    def Run(self, args):
        queues_client = GetApiAdapter(self.ReleaseTrack()).queues
        queue_ref = parsers.ParseQueue(args.queue, args.location)
        queue_short = parsers.GetConsolePromptString(queue_ref.RelativeName())
        console_io.PromptContinue(cancel_on_no=True, prompt_string='Are you sure you want to purge: [{}]'.format(queue_short))
        queues_client.Purge(queue_ref)
        log.status.Print('Purged queue [{}].'.format(queue_short))