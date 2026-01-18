from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.core import log
def add_common_list_flags(parser):
    """Inheriting from ListCommand adds flags transfer needs to modify."""
    parser.add_argument('--limit', type=arg_parsers.BoundedInt(1, sys.maxsize, unlimited=True), help='Return the first items from the API up to this limit.')
    parser.add_argument('--page-size', type=arg_parsers.BoundedInt(1, sys.maxsize, unlimited=True), default=_TRANSFER_LIST_PAGE_SIZE, help='Retrieve batches of this many items from the API.')