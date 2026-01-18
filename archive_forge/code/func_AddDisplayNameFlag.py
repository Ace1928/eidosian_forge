from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def AddDisplayNameFlag(parser):
    """Adds a --display-name flag to the given parser."""
    help_text = 'A user-friendly name for the private connection. The display name can include letters, numbers, spaces, and hyphens, and must start with a letter. The maximum length allowed is 60 characters.'
    parser.add_argument('--display-name', help=help_text, required=True)