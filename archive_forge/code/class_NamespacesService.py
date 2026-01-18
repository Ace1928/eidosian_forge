from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.run.v1 import run_v1_messages as messages
class NamespacesService(base_api.BaseApiService):
    """Service class for the namespaces resource."""
    _NAME = 'namespaces'

    def __init__(self, client):
        super(RunV1.NamespacesService, self).__init__(client)
        self._upload_configs = {}