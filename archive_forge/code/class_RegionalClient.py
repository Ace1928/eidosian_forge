from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
class RegionalClient(Client):

    @property
    def _service(self):
        return self._client.apitools_client.regionAutoscalers

    def _ScopeRequest(self, request, igm_ref):
        request.region = igm_ref.region