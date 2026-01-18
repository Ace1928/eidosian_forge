from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.netapp import flags
from googlecloudsdk.command_lib.netapp import util as netapp_util
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddActiveDirectoryUsernameArg(parser, required=True):
    """Adds a --username arg to the given parser."""
    parser.add_argument('--username', type=str, required=required, help='Username of the Active Directory domain administrator.')