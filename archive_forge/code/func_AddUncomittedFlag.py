from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
def AddUncomittedFlag(parser):
    """Adds a --uncommitted flag to the given parser."""
    help_text = 'Whether to retrieve the latest committed version of the entities or the latest version. This field is ignored if a specific commit_id is specified.'
    parser.add_argument('--uncommitted', action='store_true', help=help_text)