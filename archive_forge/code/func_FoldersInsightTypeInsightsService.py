from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
def FoldersInsightTypeInsightsService(api_version):
    """Returns the service class for the Folders insights."""
    client = RecommenderClient(api_version)
    return client.folders_locations_insightTypes_insights