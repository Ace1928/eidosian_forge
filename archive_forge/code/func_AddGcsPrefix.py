from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.database_migration.connection_profiles import flags as cp_flags
def AddGcsPrefix(parser):
    """Add the gcs prefix field to the parser."""
    parser.add_argument('--gcs-prefix', help='Cloud Storage prefix path within the bucket for the source SQL Server connection profile where the backups are stored.')