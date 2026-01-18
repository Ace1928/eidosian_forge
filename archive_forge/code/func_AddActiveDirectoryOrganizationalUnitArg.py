from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.netapp import flags
from googlecloudsdk.command_lib.netapp import util as netapp_util
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddActiveDirectoryOrganizationalUnitArg(parser):
    """Adds a --organizational-unit arg to the given parser."""
    parser.add_argument('--organizational-unit', type=str, help='The Organizational Unit (OU) within the Windows Active Directory the user belongs to.')