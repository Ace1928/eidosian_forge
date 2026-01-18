from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.database_migration import api_util
def AddCloudSQLInstanceFlag(parser, required=False):
    """Adds --cloudsql-instance flag to the given parser."""
    help_text = '    If the source or destination is a Cloud SQL database, then use this field\n    to provide the respective Cloud SQL instance ID.\n  '
    parser.add_argument('--cloudsql-instance', help=help_text, required=required)