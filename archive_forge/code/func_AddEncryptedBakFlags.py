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
def AddEncryptedBakFlags(parser):
    """Add the flags for importing encrypted BAK files.

  Add the --cert-path, --pvk-path, --pvk-password and
  --prompt-for-pvk-password flags to the parser

  Args:
    parser: The current argparse parser to add these database flags to.
  """
    enc_group = parser.add_group(mutex=False, required=False, help='Encryption info to support importing an encrypted .bak file')
    enc_group.add_argument('--cert-path', required=True, help='Path to the encryption certificate file in Google Cloud Storage associated with the BAK file. The URI is in the form `gs://bucketName/fileName`.')
    enc_group.add_argument('--pvk-path', required=True, help='Path to the encryption private key file in Google Cloud Storage associated with the BAK file. The URI is in the form `gs://bucketName/fileName`.')
    password_group = enc_group.add_group(mutex=True, required=True)
    password_group.add_argument('--pvk-password', help='The private key password associated with the BAK file.')
    password_group.add_argument('--prompt-for-pvk-password', action='store_true', help='Prompt for the private key password associated with the BAK file with character echo disabled. The password is all typed characters up to but not including the RETURN or ENTER key.')