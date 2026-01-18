from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.database_migration import api_util
from googlecloudsdk.api_lib.database_migration import conversion_workspaces
from googlecloudsdk.api_lib.database_migration import filter_rewrite
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core.resource import resource_property
import six
def _GetUpdatedMigrationJob(self, migration_job, source_ref, destination_ref, args):
    """Returns updated migration job and list of updated fields."""
    update_fields = self._GetUpdateMask(args)
    if args.IsSpecified('display_name'):
        migration_job.displayName = args.display_name
    if args.IsSpecified('type'):
        migration_job.type = self._GetType(self.messages.MigrationJob, args.type)
    if args.IsKnownAndSpecified('dump_type'):
        migration_job.dumpType = self._GetDumpType(self.messages.MigrationJob, args.dump_type)
    if args.IsSpecified('dump_path'):
        migration_job.dumpPath = args.dump_path
    if args.IsSpecified('source'):
        migration_job.source = source_ref.RelativeName()
    if args.IsSpecified('destination'):
        migration_job.destination = destination_ref.RelativeName()
    if args.IsKnownAndSpecified('dump_parallel_level'):
        migration_job.performanceConfig = self._GetPerformanceConfig(args)
    if args.IsKnownAndSpecified('filter'):
        args.filter, server_filter = filter_rewrite.Rewriter().Rewrite(args.filter)
        migration_job.filter = server_filter
    self._UpdateConnectivity(migration_job, args)
    self._UpdateLabels(args, migration_job, update_fields)
    return (migration_job, update_fields)