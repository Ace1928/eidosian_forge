from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.edge_cloud.container import util
from googlecloudsdk.api_lib.util import messages as messages_util
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
def AddLocationOptionalResourceArgForListing(parser):
    """Adds a resource argument for an Edge Container location.

  Args:
    parser: The argparse parser to add the resource arg to.
  """
    concept_parsers.ConceptParser.ForResource('--location', GetLocationsListingResourceSpec(), 'Edge Container location {}.'.format('to list'), required=False).AddToParser(parser)