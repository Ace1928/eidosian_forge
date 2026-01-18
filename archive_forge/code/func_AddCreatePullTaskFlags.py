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
def AddCreatePullTaskFlags(parser):
    """Add flags needed for creating a pull task to the parser."""
    AddQueueResourceFlag(parser, required=True)
    _GetTaskIdFlag().AddToParser(parser)
    for flag in _PullTaskFlags():
        flag.AddToParser(parser)
    _AddPayloadFlags(parser, True)