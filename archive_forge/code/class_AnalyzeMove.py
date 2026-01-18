from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.asset import client_util
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.GA)
class AnalyzeMove(base.Command):
    """Analyzes resource move."""
    detailed_help = {'DESCRIPTION': '      Analyze resource migration from its current resource hierarchy.', 'EXAMPLES': '      To analyze the impacts of moving a project to a different organization, run:\n\n          $ gcloud asset analyze-move --project=YOUR_PROJECT_ID --destination-organization=ORGANIZATION_ID\n\n      To analyze the impacts of moving a project to a different folder, run:\n\n          $ gcloud asset analyze-move --project=YOUR_PROJECT_ID --destination-folder=FOLDER_ID\n\n      To analyze only the blockers of moving a project to a different folder, run:\n\n          $ gcloud asset analyze-move --project=YOUR_PROJECT_ID --destination-folder=FOLDER_ID --blockers-only=true\n      '}

    @staticmethod
    def Args(parser):
        AddProjectArgs(parser)
        AddDestinationGroup(parser)
        AddBlockersOnlyArgs(parser)

    def Run(self, args):
        client = client_util.AnalyzeMoveClient()
        return client.AnalyzeMove(args)