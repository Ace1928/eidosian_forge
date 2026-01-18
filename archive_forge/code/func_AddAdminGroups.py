from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.gkemulticloud import util as api_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.container.gkemulticloud import constants
from googlecloudsdk.command_lib.projects import util as project_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import properties
def AddAdminGroups(parser):
    help_txt = '\nGroups of users that can perform operations as a cluster administrator.\n'
    parser.add_argument('--admin-groups', type=arg_parsers.ArgList(), metavar='GROUP', required=False, help=help_txt)