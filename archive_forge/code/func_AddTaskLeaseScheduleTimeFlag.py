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
def AddTaskLeaseScheduleTimeFlag(parser, verb):
    base.Argument('--schedule-time', required=True, help="      The task's current schedule time. This restriction is to check that the\n      caller is {} the correct task.\n      ".format(verb)).AddToParser(parser)