from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.database_migration import api_util
def AddProviderFlag(parser):
    """Adds --provider flag to the given parser."""
    help_text = '    Database provider, for managed databases.\n  '
    choices = ['RDS', 'CLOUDSQL']
    parser.add_argument('--provider', help=help_text, choices=choices)