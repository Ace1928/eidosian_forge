from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class Recommender(base.Group):
    """Manage Cloud recommendations and recommendation rules.

  Recommender allows you to retrieve recommendations for Cloud resources,
  helping you to improve security, save costs, and more. Each recommendation
  includes a suggested action, its justification, and its impact.
  Recommendations are grouped into a per-resource collection. To apply a
  recommendation, you must use the desired service's API, not the Recommender.
  Interact with and manage resources in Cloud Recommender.
  """
    category = base.API_PLATFORM_AND_ECOSYSTEMS_CATEGORY

    def Filter(self, context, args):
        base.RequireProjectID(args)
        del context, args