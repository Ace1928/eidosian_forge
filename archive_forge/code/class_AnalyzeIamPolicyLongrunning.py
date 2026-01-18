from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.asset import client_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.asset import flags
from googlecloudsdk.command_lib.asset import utils as asset_utils
from googlecloudsdk.core import log
@base.ReleaseTracks(base.ReleaseTrack.GA)
class AnalyzeIamPolicyLongrunning(base.Command):
    """Analyzes IAM policies that match a request asynchronously and writes the results to Google Cloud Storage or BigQuery destination."""
    detailed_help = DETAILED_HELP

    @staticmethod
    def Args(parser):
        flags.AddAnalyzerParentArgs(parser)
        flags.AddAnalyzerSelectorsGroup(parser)
        flags.AddAnalyzerOptionsGroup(parser, False)
        flags.AddAnalyzerConditionContextGroup(parser)
        flags.AddDestinationGroup(parser)

    def Run(self, args):
        parent = asset_utils.GetParentNameForAnalyzeIamPolicy(args.organization, args.project, args.folder)
        client = client_util.IamPolicyAnalysisLongrunningClient()
        operation = client.Analyze(parent, args)
        log.status.Print('Analyze IAM Policy in progress.')
        log.status.Print('Use [{} {}] to check the status of the operation.'.format(OPERATION_DESCRIBE_COMMAND, operation.name))