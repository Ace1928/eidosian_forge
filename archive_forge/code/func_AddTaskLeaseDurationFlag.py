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
def AddTaskLeaseDurationFlag(parser, helptext=None):
    if helptext is None:
        helptext = 'The number of seconds for the desired new lease duration, starting from now. The maximum lease duration is 1 week.'
    base.Argument('--lease-duration', required=True, type=int, help=helptext).AddToParser(parser)