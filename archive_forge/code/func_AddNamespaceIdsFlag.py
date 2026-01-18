from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import arg_parsers
def AddNamespaceIdsFlag(parser):
    """Adds flag for namespace ids to the given parser."""
    parser.add_argument('--namespace-ids', metavar='NAMESPACE_IDS', type=arg_parsers.ArgList(), help="\n      List specifying which namespaces will be included in the operation.\n      When omitted, all namespaces are included.\n\n      This is only supported for Datastore Mode databases.\n\n      For example, to operate on only the `customers` and `orders` namespaces:\n\n        $ {command} --namespaces-ids='customers','orders'\n      ")