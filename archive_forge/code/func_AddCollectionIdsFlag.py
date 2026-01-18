from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import arg_parsers
def AddCollectionIdsFlag(parser):
    """Adds flag for collection ids to the given parser.

  Args:
    parser: The argparse parser.
  """
    parser.add_argument('--collection-ids', metavar='COLLECTION_IDS', type=arg_parsers.ArgList(), help="\n      List specifying which collections will be included in the operation.\n      When omitted, all collections are included.\n\n      For example, to operate on only the `customers` and `orders`\n      collections:\n\n        $ {command} --collection-ids='customers','orders'\n      ")