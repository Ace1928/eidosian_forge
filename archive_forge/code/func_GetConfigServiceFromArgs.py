from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.recommender import service as recommender_service
from googlecloudsdk.api_lib.util import messages as messages_util
from googlecloudsdk.calliope import base
from googlecloudsdk.core import yaml
def GetConfigServiceFromArgs(api_version, is_insight_api):
    """Returns the config api service from the user-specified arguments.

  Args:
    api_version: API version string.
    is_insight_api: boolean value sepcify whether this is a insight api,
      otherwise will return a recommendation service api.
  """
    if is_insight_api:
        return recommender_service.ProjectsInsightTypeConfigsService(api_version)
    return recommender_service.ProjectsRecommenderConfigsService(api_version)