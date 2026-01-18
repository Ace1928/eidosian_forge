from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.bigtable import util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.bigtable import arguments
from googlecloudsdk.core import resources
class ListClusters(base.ListCommand):
    """List existing Bigtable clusters."""
    detailed_help = {'EXAMPLES': textwrap.dedent('          To list all clusters in an instance, run:\n\n            $ {command} --instances=my-instance-id\n\n          To list all clusters in multiple instances, run:\n\n            $ {command} --instances=my-instance-id,my-other-instance-id\n\n          ')}

    @staticmethod
    def Args(parser):
        """Register flags for this command."""
        arguments.AddInstancesResourceArg(parser, 'to list clusters for')
        parser.display_info.AddFormat('\n          table(\n            name.segment(3):sort=1:label=INSTANCE,\n            name.basename():sort=2:label=NAME,\n            location.basename():label=ZONE,\n            serveNodes:label=NODES,\n            defaultStorageType:label=STORAGE,\n            state\n          )\n        ')
        parser.display_info.AddUriFunc(_GetUriFunction)
        parser.display_info.AddCacheUpdater(arguments.InstanceCompleter)

    def Run(self, args):
        """This is what gets called when the user runs this command.

    Args:
      args: an argparse namespace. All the arguments that were provided to this
        command invocation.

    Yields:
      Some value that we want to have printed later.
    """
        cli = util.GetAdminClient()
        instance_refs = args.CONCEPTS.instances.Parse()
        if not args.IsSpecified('instances'):
            instance_refs = [util.GetInstanceRef('-')]
        for instance_ref in instance_refs:
            msg = util.GetAdminMessages().BigtableadminProjectsInstancesClustersListRequest(parent=instance_ref.RelativeName())
            for cluster in list_pager.YieldFromList(cli.projects_instances_clusters, msg, field='clusters', batch_size_attribute=None):
                yield cluster