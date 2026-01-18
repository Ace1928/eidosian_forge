from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
def FoldersRecommenderRecommendationsService(api_version):
    """Returns the service class for the Folders recommendations."""
    client = RecommenderClient(api_version)
    return client.folders_locations_recommenders_recommendations