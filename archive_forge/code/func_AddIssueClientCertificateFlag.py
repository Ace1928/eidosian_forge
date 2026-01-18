from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.api_lib.compute import constants as compute_constants
from googlecloudsdk.api_lib.container import api_adapter
from googlecloudsdk.api_lib.container import util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.container import constants
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from Google Kubernetes Engine labels that are used for the purpose of tracking
from the node pool, depending on whether locations are being added or removed.
def AddIssueClientCertificateFlag(parser):
    """Adds --issue-client-certificate flag to the parser."""
    help_text = 'Issue a TLS client certificate with admin permissions.\n\nWhen enabled, the certificate and private key pair will be present in\nMasterAuth field of the Cluster object. For cluster versions before 1.12, a\nclient certificate will be issued by default. As of 1.12, client certificates\nare disabled by default.\n'
    parser.add_argument('--issue-client-certificate', action='store_true', default=None, help=help_text)