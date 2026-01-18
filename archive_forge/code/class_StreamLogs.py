from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.ml_engine import flags
from googlecloudsdk.command_lib.ml_engine import jobs_util
from googlecloudsdk.command_lib.ml_engine import log_utils
from googlecloudsdk.core import properties
class StreamLogs(base.Command):
    """Show logs from a running AI Platform job."""
    detailed_help = DETAILED_HELP

    @staticmethod
    def Args(parser):
        _AddStreamLogsArgs(parser)
        parser.display_info.AddFormat(log_utils.LOG_FORMAT)

    def Run(self, args):
        """Run the stream-logs command."""
        return jobs_util.StreamLogs(args.job, args.task_name, properties.VALUES.ml_engine.polling_interval.GetInt(), args.allow_multiline_logs)