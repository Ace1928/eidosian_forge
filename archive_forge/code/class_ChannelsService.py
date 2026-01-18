import platform
import sys
from apitools.base.py import base_api
import gslib
from gslib.metrics import MetricsCollector
from gslib.third_party.storage_apitools import storage_v1_messages as messages
class ChannelsService(base_api.BaseApiService):
    """Service class for the channels resource."""
    _NAME = u'channels'

    def __init__(self, client):
        super(StorageV1.ChannelsService, self).__init__(client)
        self._upload_configs = {}

    def Stop(self, request, global_params=None):
        """Stop watching resources through this channel.

      Args:
        request: (Channel) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (StorageChannelsStopResponse) The response message.
      """
        config = self.GetMethodConfig('Stop')
        return self._RunMethod(config, request, global_params=global_params)
    Stop.method_config = lambda: base_api.ApiMethodInfo(http_method=u'POST', method_id=u'storage.channels.stop', ordered_params=[], path_params=[], query_params=[], relative_path=u'channels/stop', request_field='<request>', request_type_name=u'Channel', response_type_name=u'StorageChannelsStopResponse', supports_download=False)