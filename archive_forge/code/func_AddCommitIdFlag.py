from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.calliope import arg_parsers
def AddCommitIdFlag(parser):
    """Adds a --commit-id flag to the given parser."""
    help_text = 'Commit id for the conversion workspace to use for creating the migration job. If not specified, the latest commit id will be used by default.'
    parser.add_argument('--commit-id', help=help_text)