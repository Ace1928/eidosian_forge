from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.database_migration import api_util
from googlecloudsdk.api_lib.database_migration import connection_profiles
from googlecloudsdk.api_lib.database_migration import resource_args
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.database_migration import flags
from googlecloudsdk.command_lib.database_migration.connection_profiles import flags as cp_flags
from googlecloudsdk.core.console import console_io
@base.ReleaseTracks(base.ReleaseTrack.GA)
class UpdateGA(_Update, base.Command):
    """Update a Database Migration Service connection profile."""

    @staticmethod
    def Args(parser):
        _Update.Args(parser)
        cp_flags.AddClientCertificateFlag(parser)
        cp_flags.AddCloudSQLInstanceFlag(parser)
        cp_flags.AddAlloydbClusterFlag(parser)