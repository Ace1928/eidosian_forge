from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.recommender import base
from googlecloudsdk.api_lib.recommender import flag_utils
class InsightTypes(base.ClientBase):
    """Base client to list Insight Types for all versions."""

    def __init__(self, api_version):
        super(InsightTypes, self).__init__(api_version)
        self._service = self._client.insightTypes

    def List(self, page_size, limit=None):
        """List Insight Types.

    Args:
      page_size: int, The number of items to retrieve per request.
      limit: int, The maximum number of records to yield.

    Returns:
      The list of insight types.
    """
        request = self._messages.RecommenderInsightTypesListRequest()
        return list_pager.YieldFromList(self._service, request, batch_size_attribute='pageSize', batch_size=page_size, limit=limit, field='insightTypes')