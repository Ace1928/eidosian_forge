from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.database_migration import api_util
def AddPortFlag(parser, required=False):
    """Adds --port flag to the given parser."""
    help_text = '    Network port of the database.\n  '
    parser.add_argument('--port', help=help_text, required=required, type=int)