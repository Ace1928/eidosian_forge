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
def ParseExtensionConstraints(args):
    """Parse extension constraints flags into CertificateExtensionConstraints API message.

  Assumes that the parser defined by args has the flags
  copy_all_requested_extensions, copy_known_extesnions, and
  copy-extensions-by-oid. Also supports drop_known_extensions and
  drop_oid_extensions for clearing the extension lists.

  Args:
    args: The argparse object to read flags from.

  Returns:
    The CertificateExtensionConstraints API message.
  """
    if args.IsSpecified('copy_all_requested_extensions'):
        return None
    messages = privateca_base.GetMessagesModule('v1')
    known_exts = []
    if not args.IsKnownAndSpecified('drop_known_extensions') and args.IsSpecified('copy_known_extensions'):
        known_exts = [_StrToKnownExtension('--copy-known-extensions', ext) for ext in args.copy_known_extensions]
    oids = []
    if not args.IsKnownAndSpecified('drop_oid_extensions') and args.IsSpecified('copy_extensions_by_oid'):
        oids = args.copy_extensions_by_oid
    return messages.CertificateExtensionConstraints(knownExtensions=known_exts, additionalExtensions=oids)