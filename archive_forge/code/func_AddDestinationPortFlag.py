from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def AddDestinationPortFlag(parser):
    """Adds a --display-name flag to the given parser."""
    help_text = 'Destination port for connection.'
    parser.add_argument('--destination-port', help=help_text, type=int)