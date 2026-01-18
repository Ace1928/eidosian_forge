from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.tasks import GetApiAdapter
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.tasks import constants
from googlecloudsdk.command_lib.tasks import flags
from googlecloudsdk.command_lib.tasks import parsers
from googlecloudsdk.core import log
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class CreateHttp(base.CreateCommand):
    """Create and add a task that targets a HTTP endpoint."""
    detailed_help = {'DESCRIPTION': '          {description}\n          ', 'EXAMPLES': '          To create a task:\n\n              $ {command} --queue=my-queue\n                --url=http://example.com/handler-path my-task\n         '}

    @staticmethod
    def Args(parser):
        flags.AddCreateHttpTaskFlags(parser)
        flags.AddLocationFlag(parser)

    def Run(self, args):
        if self.ReleaseTrack() == base.ReleaseTrack.ALPHA:
            api_release_track = base.ReleaseTrack.BETA
        else:
            api_release_track = self.ReleaseTrack()
        api = GetApiAdapter(api_release_track)
        tasks_client = api.tasks
        queue_ref = parsers.ParseQueue(args.queue, args.location)
        task_ref = parsers.ParseTask(args.task, queue_ref) if args.task else None
        task_config = parsers.ParseCreateTaskArgs(args, constants.HTTP_TASK, api.messages, release_track=api_release_track)
        create_response = tasks_client.Create(queue_ref, task_ref, schedule_time=task_config.scheduleTime, http_request=task_config.httpRequest)
        log.CreatedResource(create_response.name, 'task')
        return create_response