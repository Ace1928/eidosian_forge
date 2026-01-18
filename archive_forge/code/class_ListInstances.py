from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.bigtable import util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.bigtable import arguments
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
class ListInstances(base.ListCommand):
    """List existing Bigtable instance tables.

  ## EXAMPLES
  To list all tables in an instance, run:

    $ {command} --instances=INSTANCE_NAME

  To list all tables in several instances, run:
    $ {command} --instances=INSTANCE_NAME1,INSTANCE_NAME2
  """

    @staticmethod
    def Args(parser):
        """Register flags for this command."""
        parser.display_info.AddFormat('\n          table(\n            name.basename():sort=1\n          )\n        ')
        parser.display_info.AddUriFunc(_GetUriFunction)
        arguments.ArgAdder(parser).AddInstance(positional=False, required=True, multiple=True)

    def Run(self, args):
        cli = util.GetAdminClient()
        msgs = util.GetAdminMessages()
        instances = args.instances
        results = []
        for instance in instances:
            instance_ref = resources.REGISTRY.Parse(instance, params={'projectsId': properties.VALUES.core.project.GetOrFail}, collection='bigtableadmin.projects.instances')
            request = msgs.BigtableadminProjectsInstancesTablesListRequest(parent=instance_ref.RelativeName())
            for table in list_pager.YieldFromList(cli.projects_instances_tables, request, field='tables', batch_size_attribute=None):
                results.append(table)
        return results