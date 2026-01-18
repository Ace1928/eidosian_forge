from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.ml_engine import flags
from googlecloudsdk.command_lib.ml_engine import jobs_util
from googlecloudsdk.command_lib.ml_engine import log_utils
from googlecloudsdk.core import properties
def _AddStreamLogsArgs(parser):
    flags.JOB_NAME.AddToParser(parser)
    flags.POLLING_INTERVAL.AddToParser(parser)
    flags.ALLOW_MULTILINE_LOGS.AddToParser(parser)
    flags.TASK_NAME.AddToParser(parser)