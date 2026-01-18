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
def AddServerCertificatesFilesFlag(parser, required=False):
    parser.add_argument('--server-certificates-files', type=arg_parsers.ArgList(), metavar='SERVER_CERTIFICATES', help='A list of filenames of leaf server certificates used to authenticate HTTPS connections to the EKM replica in PEM format. If files are not in PEM, the assumed format will be DER.', required=required)