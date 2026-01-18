from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddAsnArg(parser):
    """Adds Asn argument for routers."""
    parser.add_argument('--asn', required=False, type=int, help='The optional BGP autonomous system number (ASN) for this router. Must be a 16-bit or 32-bit private ASN as defined in https://tools.ietf.org/html/rfc6996, for example `--asn=64512`.')