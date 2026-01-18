from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import itertools
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.recommender import locations
from googlecloudsdk.api_lib.recommender import recommendation
from googlecloudsdk.api_lib.recommender import recommenders
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.recommender import flags
@base.ReleaseTracks(base.ReleaseTrack.GA)
class ListOriginal(base.ListCommand):
    """List operations for a recommendation.

  This command lists all recommendations for a given Google Cloud entity ID,
  location, and recommender. Supported recommenders can be found here:
  https://cloud.google.com/recommender/docs/recommenders.
  The following Google Cloud entity types are supported: project,
  billing_account, folder and organization.
  """
    detailed_help = DETAILED_HELP

    @staticmethod
    def Args(parser):
        """Args is called by calliope to gather arguments for this command.

    Args:
      parser: An argparse parser that you can use to add arguments that go on
        the command line after this command.
    """
        flags.AddParentFlagsToParser(parser)
        parser.add_argument('--location', metavar='LOCATION', required=True, help='Location to list recommendations for.')
        parser.add_argument('--recommender', metavar='RECOMMENDER', required=True, help='Recommender to list recommendations for. Supported recommenders can be found here: https://cloud.google.com/recommender/docs/recommenders.')
        parser.display_info.AddFormat(DISPLAY_FORMAT)

    def Run(self, args):
        """Run 'gcloud recommender recommendations list'.

    Args:
      args: argparse.Namespace, The arguments that this command was invoked
        with.

    Returns:
      The list of recommendations for this project.
    """
        recommendations_client = recommendation.CreateClient(self.ReleaseTrack())
        parent_name = flags.GetRecommenderName(args)
        return recommendations_client.List(parent_name, args.page_size)