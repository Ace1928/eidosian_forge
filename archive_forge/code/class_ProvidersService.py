from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.containeranalysis.v1alpha1 import containeranalysis_v1alpha1_messages as messages
class ProvidersService(base_api.BaseApiService):
    """Service class for the providers resource."""
    _NAME = 'providers'

    def __init__(self, client):
        super(ContaineranalysisV1alpha1.ProvidersService, self).__init__(client)
        self._upload_configs = {}