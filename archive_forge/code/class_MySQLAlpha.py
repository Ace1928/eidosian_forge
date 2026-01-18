from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.database_migration import resource_args
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.database_migration import flags
from googlecloudsdk.command_lib.database_migration.connection_profiles import create_helper
from googlecloudsdk.command_lib.database_migration.connection_profiles import flags as cp_flags
from googlecloudsdk.core.console import console_io
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class MySQLAlpha(_MySQL, base.Command):
    """Create a Database Migration Service connection profile for MySQL."""
    detailed_help = {'DESCRIPTION': DESCRIPTION, 'EXAMPLES': EXAMPLES.format(instance='instance')}

    @staticmethod
    def Args(parser):
        _MySQL.Args(parser)
        cp_flags.AddSslConfigGroup(parser, base.ReleaseTrack.ALPHA)
        cp_flags.AddInstanceFlag(parser)