from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import ipaddress
import re
from googlecloudsdk.api_lib.privateca import base as privateca_base
from googlecloudsdk.api_lib.util import messages as messages_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.privateca import exceptions as privateca_exceptions
from googlecloudsdk.command_lib.privateca import preset_profiles
from googlecloudsdk.command_lib.privateca import text_utils
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import times
import six
from that bucket.
def AddIdentityConstraintsFlags(parser, require_passthrough_flags=True):
    """Adds flags for expressing identity constraints.

  Args:
    parser: The argparse object to add the flags to.
    require_passthrough_flags: Whether the boolean --copy-* flags should be
      required.
  """
    base.Argument('--identity-cel-expression', help='A CEL expression that will be evaluated against the identity in the certificate before it is issued, and returns a boolean signifying whether the request should be allowed.').AddToParser(parser)
    base.Argument('--copy-subject', help='If this is specified, the Subject from the certificate request will be copied into the signed certificate. Specify --no-copy-subject to drop any caller-specified subjects from the certificate request.', action='store_true', required=require_passthrough_flags).AddToParser(parser)
    base.Argument('--copy-sans', help='If this is specified, the Subject Alternative Name extension from the certificate request will be copied into the signed certificate. Specify --no-copy-sans to drop any caller-specified SANs in the certificate request.', action='store_true', required=require_passthrough_flags).AddToParser(parser)