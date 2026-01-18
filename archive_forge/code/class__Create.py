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
class _Create(object):
    """Create a Database Migration Service migration job."""

    @staticmethod
    def Args(parser):
        """Args is called by calliope to gather arguments for this command.

    Args:
      parser: An argparse parser that you can use to add arguments that go on
        the command line after this command. Positional arguments are allowed.
    """
        mj_flags.AddNoAsyncFlag(parser)
        mj_flags.AddDisplayNameFlag(parser)
        mj_flags.AddTypeFlag(parser, required=True)
        mj_flags.AddDumpPathFlag(parser)
        mj_flags.AddConnectivityGroupFlag(parser, mj_flags.ApiType.CREATE, required=True)
        flags.AddLabelsCreateFlags(parser)

    def Run(self, args):
        """Create a Database Migration Service migration job.

    Args:
      args: argparse.Namespace, The arguments that this command was invoked
        with.

    Returns:
      A dict object representing the operations resource describing the create
      operation if the create was successful.
    """
        migration_job_ref = args.CONCEPTS.migration_job.Parse()
        parent_ref = migration_job_ref.Parent().RelativeName()
        source_ref = args.CONCEPTS.source.Parse()
        destination_ref = args.CONCEPTS.destination.Parse()
        if self.ReleaseTrack() == base.ReleaseTrack.GA:
            conversion_workspace_ref = args.CONCEPTS.conversion_workspace.Parse()
            cmek_key_ref = args.CONCEPTS.cmek_key.Parse()
        else:
            conversion_workspace_ref = None
            cmek_key_ref = None
        mj_client = migration_jobs.MigrationJobsClient(self.ReleaseTrack())
        result_operation = mj_client.Create(parent_ref, migration_job_ref.migrationJobsId, source_ref, destination_ref, conversion_workspace_ref, cmek_key_ref, args)
        client = api_util.GetClientInstance(self.ReleaseTrack())
        messages = api_util.GetMessagesModule(self.ReleaseTrack())
        resource_parser = api_util.GetResourceParser(self.ReleaseTrack())
        if args.IsKnownAndSpecified('no_async'):
            log.status.Print('Waiting for migration job [{}] to be created with [{}]'.format(migration_job_ref.migrationJobsId, result_operation.name))
            api_util.HandleLRO(client, result_operation, client.projects_locations_migrationJobs)
            log.status.Print('Created migration job {} [{}]'.format(migration_job_ref.migrationJobsId, result_operation.name))
            return
        operation_ref = resource_parser.Create('datamigration.projects.locations.operations', operationsId=result_operation.name, projectsId=migration_job_ref.projectsId, locationsId=migration_job_ref.locationsId)
        return client.projects_locations_operations.Get(messages.DatamigrationProjectsLocationsOperationsGetRequest(name=operation_ref.operationsId))