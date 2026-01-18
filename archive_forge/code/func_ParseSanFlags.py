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
def ParseSanFlags(args):
    """Validates the san flags and creates a SubjectAltNames message from them.

  Args:
    args: The parser that contains the flags.

  Returns:
    The SubjectAltNames message with the flag data.
  """
    email_addresses, dns_names, ip_addresses, uris = ([], [], [], [])
    if args.IsSpecified('email_san'):
        email_addresses = list(map(ValidateEmailSanFlag, args.email_san))
    if args.IsSpecified('dns_san'):
        dns_names = list(map(ValidateDnsSanFlag, args.dns_san))
    if args.IsSpecified('ip_san'):
        ip_addresses = list(map(ValidateIpSanFlag, args.ip_san))
    if args.IsSpecified('uri_san'):
        uris = args.uri_san
    return privateca_base.GetMessagesModule('v1').SubjectAltNames(emailAddresses=email_addresses, dnsNames=dns_names, ipAddresses=ip_addresses, uris=uris)