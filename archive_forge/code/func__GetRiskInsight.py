from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.recommender import insight
from googlecloudsdk.command_lib.projects import util as project_util
def _GetRiskInsight(release_track, project_id, insight_type, request_filter=None, matcher=None):
    """Returns the first insight fetched by the recommender API.

  Args:
    release_track: Release track of the recommender.
    project_id: Project ID.
    insight_type: String insight type.
    request_filter: Optional string filter for the recommender.
    matcher: Matcher for the insight object. None means match all.

  Returns:
    Insight object returned by the recommender API. Returns 'None' if no
    matching insights were found. Returns the first insight object that matches
    the matcher. If no matcher, returns the first insight object fetched.
  """
    client = insight.CreateClient(release_track)
    parent_name = 'projects/{0}/locations/global/insightTypes/{1}'.format(project_id, insight_type)
    result = client.List(parent_name, page_size=100, limit=None, request_filter=request_filter)
    for r in result:
        if not matcher:
            return r
        if matcher(r):
            return r
    return None