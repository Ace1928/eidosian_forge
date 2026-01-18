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
def AddExtensionConstraintsFlags(parser):
    """Adds flags for expressing extension constraints.

  Args:
    parser: The argparser to add the arguments to.
  """
    extension_group = parser.add_group(mutex=True, required=False, help='Constraints on requested X.509 extensions. If unspecified, all extensions from certificate request will be ignored when signing the certificate.')
    copy_group = extension_group.add_group(mutex=False, required=False, help='Specify exact x509 extensions to copy by OID or known extension.')
    base.Argument('--copy-extensions-by-oid', help='If this is set, then extensions with the given OIDs will be copied from the certificate request into the signed certificate.', type=arg_parsers.ArgList(element_type=_StrToObjectId), metavar='OBJECT_ID').AddToParser(copy_group)
    known_extensions = GetKnownExtensionMapping()
    base.Argument('--copy-known-extensions', help='If this is set, then the given extensions will be copied from the certificate request into the signed certificate.', type=arg_parsers.ArgList(choices=known_extensions, visible_choices=[ext for ext in known_extensions.keys() if ext not in _HIDDEN_KNOWN_EXTENSIONS]), metavar='KNOWN_EXTENSIONS').AddToParser(copy_group)
    base.Argument('--copy-all-requested-extensions', help='If this is set, all extensions specified in the certificate  request will be copied into the signed certificate.', action='store_const', const=True).AddToParser(extension_group)