from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.api_lib.bigtable import util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.bigtable import arguments
class DescribeCluster(base.DescribeCommand):
    """Describe an existing Bigtable cluster."""
    detailed_help = {'EXAMPLES': textwrap.dedent("          To view a cluster's description, run:\n\n            $ {command} my-cluster-id --instance=my-instance-id\n\n          ")}

    @staticmethod
    def Args(parser):
        """Register flags for this command."""
        arguments.AddClusterResourceArg(parser, 'to describe')

    def Run(self, args):
        """This is what gets called when the user runs this command.

    Args:
      args: an argparse namespace. All the arguments that were provided to this
        command invocation.

    Returns:
      Some value that we want to have printed later.
    """
        cli = util.GetAdminClient()
        cluster_ref = args.CONCEPTS.cluster.Parse()
        msg = util.GetAdminMessages().BigtableadminProjectsInstancesClustersGetRequest(name=cluster_ref.RelativeName())
        return cli.projects_instances_clusters.Get(msg)