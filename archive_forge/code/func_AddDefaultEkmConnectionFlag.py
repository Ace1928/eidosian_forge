from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.kms import maps
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util import parameter_info_lib
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import times
def AddDefaultEkmConnectionFlag(parser, required=False):
    parser.add_argument('--default-ekm-connection', help='The resource name of the EkmConnection to be used as the default EkmConnection for all `external-vpc` CryptoKeys in a project and location. Can be an empty string to remove the default EkmConnection.', required=required)