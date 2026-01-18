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
def _GetTaskIdFlag():
    return base.Argument('task', metavar='TASK_ID', nargs='?', help="      The task to create.\n\n      If not specified then the system will generate a random unique task\n      ID. Explicitly specifying a task ID enables task de-duplication. If a\n      task's ID is identical to that of an existing task or a task that was\n      deleted or completed recently then the call will fail.\n\n      Because there is an extra lookup cost to identify duplicate task\n      names, tasks created with IDs have significantly increased latency.\n      Using hashed strings for the task ID or for the prefix of the task ID\n      is recommended.\n      ")