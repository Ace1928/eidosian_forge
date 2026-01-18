from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.tasks import GetApiAdapter
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.tasks import flags
from googlecloudsdk.command_lib.tasks import parsers
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class RenewLease(base.Command):
    """Renew the lease on a task in a pull queue."""

    @staticmethod
    def Args(parser):
        flags.AddTaskResourceArgs(parser, 'to renew the lease of')
        flags.AddLocationFlag(parser)
        flags.AddTaskLeaseScheduleTimeFlag(parser, 'renewing')
        flags.AddTaskLeaseDurationFlag(parser)

    def Run(self, args):
        tasks_client = GetApiAdapter(self.ReleaseTrack()).tasks
        queue_ref = parsers.ParseQueue(args.queue, args.location)
        task_ref = parsers.ParseTask(args.task, queue_ref)
        duration = parsers.FormatLeaseDuration(args.lease_duration)
        return tasks_client.RenewLease(task_ref, args.schedule_time, duration)