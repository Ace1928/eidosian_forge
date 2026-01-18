from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def add_encryption_flags(parser, allow_patch=False, command_only_reads_data=False, hidden=False):
    """Adds flags for encryption and decryption keys.

  Args:
    parser (parser_arguments.ArgumentInterceptor): Parser passed to surface.
    allow_patch (bool): Adds flags relevant for update operations if true.
    command_only_reads_data (bool): Should be set to true if a command only
        reads data from storage providers (e.g. cat, ls) and false for commands
        that also write data (e.g. cp, rewrite). Hides flags that pertain to
        write operations for read-only commands.
    hidden (bool): Hides encryption flags if true.
  """
    encryption_group = parser.add_group(category='ENCRYPTION', hidden=hidden)
    encryption_group.add_argument('--encryption-key', hidden=hidden or command_only_reads_data, help='The encryption key to use for encrypting target objects. The specified encryption key can be a customer-supplied encryption key (An RFC 4648 section 4 base64-encoded AES256 string), or a customer-managed encryption key of the form `projects/{project}/locations/{location}/keyRings/{key-ring}/cryptoKeys/{crypto-key}`. The specified key also acts as a decryption key, which is useful when copying or moving encrypted data to a new location. Using this flag in an `objects update` command triggers a rewrite of target objects.')
    encryption_group.add_argument('--decryption-keys', type=arg_parsers.ArgList(), metavar='DECRYPTION_KEY', hidden=hidden, help='A comma-separated list of customer-supplied encryption keys (RFC 4648 section 4 base64-encoded AES256 strings) that will be used to decrypt Cloud Storage objects. Data encrypted with a customer-managed encryption key (CMEK) is decrypted automatically, so CMEKs do not need to be listed here.')
    if allow_patch:
        encryption_group.add_argument('--clear-encryption-key', action='store_true', hidden=hidden or command_only_reads_data, help='Clears the encryption key associated with an object. Using this flag triggers a rewrite of affected objects, which are then encrypted using the default encryption key set on the bucket, if one exists, or else with a Google-managed encryption key.')