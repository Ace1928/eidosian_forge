from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.gkemulticloud import util as api_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.container.gkemulticloud import constants
from googlecloudsdk.command_lib.projects import util as project_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import properties
def AddEnableAutoRepair(parser, for_create=False):
    help_text = 'Enable node autorepair feature for a node pool. Use --no-enable-autorepair to disable.\n\n  $ {command} --enable-autorepair\n'
    if for_create:
        help_text += '\nNode autorepair is disabled by default.\n'
    parser.add_argument('--enable-autorepair', action='store_true', default=None, help=help_text)