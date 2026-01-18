from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.recommender import base
from googlecloudsdk.api_lib.recommender import flag_utils
def _CreateMarkRequest(self, name, state, state_metadata, etag):
    """Creates MarkRequest with the specified state."""
    request_name = 'MarkRecommendation{}Request'.format(state)
    mark_request = self._GetVersionedMessage(request_name)(etag=etag)
    if state_metadata:
        metadata = encoding.DictToAdditionalPropertyMessage(state_metadata, self._GetVersionedMessage(request_name).StateMetadataValue, sort_items=True)
        mark_request.stateMetadata = metadata
    kwargs = {'name': name, flag_utils.ToCamelCase(self._message_prefix + request_name): mark_request}
    return self._GetMessage('RecommenderProjectsLocationsRecommendersRecommendationsMark{}Request'.format(state))(**kwargs)