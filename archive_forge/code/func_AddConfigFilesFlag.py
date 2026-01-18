from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
def AddConfigFilesFlag(parser):
    """Adds a --config-files flag to the given parser."""
    help_text = '    A list of files to import rules from. Either provide a single file path or if\n    multiple files are to be provided, each file should correspond to one schema.\n    Provide file paths as a comma separated list.\n    '
    parser.add_argument('--config-files', metavar='CONGIF_FILE', type=arg_parsers.ArgList(min_length=1), help=help_text, required=True)