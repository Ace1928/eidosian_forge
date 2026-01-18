from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.database_migration import resource_args
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.database_migration import flags
from googlecloudsdk.command_lib.database_migration.connection_profiles import cloudsql_flags as cs_flags
from googlecloudsdk.command_lib.database_migration.connection_profiles import create_helper
from googlecloudsdk.command_lib.database_migration.connection_profiles import flags as cp_flags
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class CloudSQLAlpha(_CloudSQL, base.Command):
    """Create a Database Migration Service connection profile for Cloud SQL."""
    detailed_help = DETAILED_HELP

    @staticmethod
    def Args(parser):
        _CloudSQL.Args(parser)
        cs_flags.AddDatabaseVersionFlag(parser, support_minor_version=False)