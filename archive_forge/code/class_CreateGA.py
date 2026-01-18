from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.database_migration import api_util
from googlecloudsdk.api_lib.database_migration import migration_jobs
from googlecloudsdk.api_lib.database_migration import resource_args
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.database_migration import flags
from googlecloudsdk.command_lib.database_migration.migration_jobs import flags as mj_flags
from googlecloudsdk.core import log
@base.ReleaseTracks(base.ReleaseTrack.GA)
class CreateGA(_Create, base.Command):
    """Create a Database Migration Service migration job."""
    detailed_help = DETAILED_HELP_GA

    @staticmethod
    def Args(parser):
        """Args is called by calliope to gather arguments for this command.

    Args:
      parser: An argparse parser that you can use to add arguments that go on
        the command line after this command. Positional arguments are allowed.
    """
        resource_args.AddHeterogeneousMigrationJobResourceArgs(parser, 'to create', required=True)
        _Create.Args(parser)
        mj_flags.AddFilterFlag(parser)
        mj_flags.AddCommitIdFlag(parser)
        mj_flags.AddDumpParallelLevelFlag(parser)
        mj_flags.AddSqlServerHomogeneousMigrationConfigFlag(parser)
        mj_flags.AddDumpTypeFlag(parser)