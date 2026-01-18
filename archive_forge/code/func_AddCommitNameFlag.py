from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
def AddCommitNameFlag(parser):
    """Adds a --commit-name flag to the given parser."""
    help_text = '\n    A user-friendly name for the conversion workspace commit. The commit name\n    can include letters, numbers, spaces, and hyphens, and must start with a\n    letter.\n    '
    parser.add_argument('--commit-name', help=help_text)