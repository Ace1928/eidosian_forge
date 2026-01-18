from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.bigtable import util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.bigtable import arguments
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
class ListOperations(base.ListCommand):
    """List Cloud Bigtable operations."""
    detailed_help = {'DESCRIPTION': textwrap.dedent('          List Cloud Bigtable operations.\n          '), 'EXAMPLES': textwrap.dedent('          To list all operations for the default project, run:\n\n            $ {command}\n\n          To list all operations for instance INSTANCE_NAME, run:\n\n            $ {command} --instance=INSTANCE_NAME\n          ')}

    @staticmethod
    def Args(parser):
        """Register flags for this command."""
        arguments.AddInstanceResourceArg(parser, 'to list operations for', required=False)
        parser.display_info.AddFormat('\n          table(\n             name():label=NAME,\n             done,\n             metadata.firstof(startTime, requestTime, progress.start_time).date():label=START_TIME:sort=1:reverse,\n             metadata.firstof(endTime, finishTime, progress.end_time).date():label=END_TIME\n           )\n        ')
        parser.display_info.AddUriFunc(_GetUriFunction)
        parser.display_info.AddTransforms({'name': _TransformOperationName})

    def Run(self, args):
        """This is what gets called when the user runs this command.

    Args:
      args: an argparse namespace. All the arguments that were provided to this
        command invocation.

    Returns:
      Some value that we want to have printed later.
    """
        cli = util.GetAdminClient()
        ref_name = 'operations/' + resources.REGISTRY.Parse(properties.VALUES.core.project.Get(required=True), collection='bigtableadmin.projects').RelativeName()
        if args.IsSpecified('instance'):
            ref_name = ref_name + '/instances/' + args.instance
        msg = util.GetAdminMessages().BigtableadminOperationsProjectsOperationsListRequest(name=ref_name)
        return list_pager.YieldFromList(cli.operations_projects_operations, msg, field='operations', batch_size_attribute=None)