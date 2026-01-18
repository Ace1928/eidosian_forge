from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from googlecloudsdk.api_lib.compute import utils as compute_utils
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util import completers
def AddSslMode(parser, hidden=False):
    """Adds the '--ssl-mode' flag to the parser.

  Args:
    parser: The current argparse parser to add this to.
    hidden: if the field needs to be hidden.
  """
    help_text = 'Set the SSL mode of the instance.'
    parser.add_argument('--ssl-mode', choices={'ALLOW_UNENCRYPTED_AND_ENCRYPTED': 'Allow non-SSL and SSL connections. For SSL connections, client certificate will not be verified.', 'ENCRYPTED_ONLY': 'Only allow connections encrypted with SSL/TLS.', 'TRUSTED_CLIENT_CERTIFICATE_REQUIRED': 'Only allow connections encrypted with SSL/TLS and with valid client certificates.'}, required=False, default=None, help=help_text, hidden=hidden)