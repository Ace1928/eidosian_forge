from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddStorageAutoResizeLimitFlag(parser):
    """Adds a --storage-auto-resize-limit flag to the given parser."""
    help_text = '    Maximum size to which storage capacity can be automatically increased. The\n    default value is 0, which specifies that there is no limit.\n    '
    parser.add_argument('--storage-auto-resize-limit', type=int, help=help_text)