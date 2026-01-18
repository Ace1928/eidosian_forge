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
def AddExtensionConstraintsFlagsForUpdate(parser):
    """Adds flags for updating extension constraints.

  Args:
    parser: The argparser to add the arguments to.
  """
    extension_group_help = 'Constraints on requested X.509 extensions.'
    extension_group = parser.add_group(mutex=True, required=False, help=extension_group_help)
    copy_group = extension_group.add_group(mutex=False, required=False, help='Specify exact x509 extensions to copy by OID or known extension.')
    oid_group = copy_group.add_group(mutex=True, required=False, help='Constraints on unknown extensions by their OIDs.')
    base.Argument('--copy-extensions-by-oid', help='If this is set, then extensions with the given OIDs will be copied from the certificate request into the signed certificate.', type=arg_parsers.ArgList(element_type=_StrToObjectId), metavar='OBJECT_ID').AddToParser(oid_group)
    base.Argument('--drop-oid-extensions', help='If this is set, then all existing OID extensions will be removed from the template, prohibiting any extensions specified by OIDs to be specified by the requester.', action='store_const', const=True).AddToParser(oid_group)
    known_group = copy_group.add_group(mutex=True, required=False, help='Constraints on known extensions.')
    known_extensions = GetKnownExtensionMapping()
    base.Argument('--copy-known-extensions', help='If this is set, then the given extensions will be copied from the certificate request into the signed certificate.', type=arg_parsers.ArgList(choices=known_extensions, visible_choices=[ext for ext in known_extensions.keys() if ext not in _HIDDEN_KNOWN_EXTENSIONS]), metavar='KNOWN_EXTENSIONS').AddToParser(known_group)
    base.Argument('--drop-known-extensions', help='If this is set, then all known extensions will be removed from the template, prohibiting any known x509 extensions to be specified by the requester.', action='store_const', const=True).AddToParser(known_group)
    base.Argument('--copy-all-requested-extensions', help='If this is set, all extensions, whether known or specified by OID, that are specified in the certificate request will be copied into the signed certificate.', action='store_const', const=True).AddToParser(extension_group)