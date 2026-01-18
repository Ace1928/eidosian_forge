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
def AddQueueResourceFlag(parser, required=True, plural_tasks=False):
    description = 'The queue the tasks belong to.' if plural_tasks else 'The queue the task belongs to.'
    argument = base.Argument('--queue', help=description, required=required)
    argument.AddToParser(parser)