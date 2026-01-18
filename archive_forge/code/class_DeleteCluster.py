from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.api_lib.bigtable import clusters
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.bigtable import arguments
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
class DeleteCluster(base.DeleteCommand):
    """Delete a bigtable cluster."""
    detailed_help = {'EXAMPLES': textwrap.dedent('          To delete a cluster, run:\n\n            $ {command} my-cluster-id --instance=my-instance-id\n\n          ')}

    @staticmethod
    def Args(parser):
        """Register flags for this command."""
        arguments.AddClusterResourceArg(parser, 'to delete')

    def Run(self, args):
        """This is what gets called when the user runs this command.

    Args:
      args: an argparse namespace. All the arguments that were provided to this
        command invocation.

    Returns:
      Some value that we want to have printed later.
    """
        cluster_ref = args.CONCEPTS.cluster.Parse()
        console_io.PromptContinue('You are about to delete cluster: [{0}]'.format(cluster_ref.Name()), throw_if_unattended=True, cancel_on_no=True)
        response = clusters.Delete(cluster_ref)
        log.DeletedResource(cluster_ref.Name(), 'cluster')
        return response