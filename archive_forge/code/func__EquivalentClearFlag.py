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
def _EquivalentClearFlag(flag):
    name = flag.name.replace('--', '--clear-')
    clear_flag = base.Argument(name, action='store_true', help='      Clear the field corresponding to `{}`.'.format(flag.name))
    return clear_flag