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
def _PullQueueFlags():
    return [base.Argument('--max-attempts', type=arg_parsers.BoundedInt(-1, sys.maxsize, unlimited=True), help='          The maximum number of attempts per task in the queue.\n          '), base.Argument('--max-retry-duration', help='          The time limit for retrying a failed task, measured from when the task\n          was first run. Once the `--max-retry-duration` time has passed and the\n          task has been attempted --max-attempts times, no further attempts will\n          be made and the task will be deleted.\n\n          Must be a string that ends in \'s\', such as "5s".\n          ')]