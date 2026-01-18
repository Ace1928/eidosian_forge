from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.gkemulticloud import util as api_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.container.gkemulticloud import constants
from googlecloudsdk.command_lib.projects import util as project_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import properties
def AddAdminUsers(parser, create=True):
    help_txt = 'Users that can perform operations as a cluster administrator.'
    if create:
        help_txt += ' If not specified, the value of property core/account is used.'
    parser.add_argument('--admin-users', type=arg_parsers.ArgList(min_length=1), metavar='USER', help=help_txt)