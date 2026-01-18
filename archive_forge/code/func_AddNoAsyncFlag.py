from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def AddNoAsyncFlag(parser):
    """Adds a --no-async flag to the given parser."""
    help_text = 'Waits for the operation in progress to complete before returning.'
    parser.add_argument('--no-async', action='store_true', help=help_text)