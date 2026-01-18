from __future__ import absolute_import
from apitools.base.py import base_api
from samples.fusiontables_sample.fusiontables_v1 import fusiontables_v1_messages as messages
class StyleService(base_api.BaseApiService):
    """Service class for the style resource."""
    _NAME = u'style'

    def __init__(self, client):
        super(FusiontablesV1.StyleService, self).__init__(client)
        self._upload_configs = {}

    def Delete(self, request, global_params=None):
        """Deletes a style.

      Args:
        request: (FusiontablesStyleDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (FusiontablesStyleDeleteResponse) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method=u'DELETE', method_id=u'fusiontables.style.delete', ordered_params=[u'tableId', u'styleId'], path_params=[u'styleId', u'tableId'], query_params=[], relative_path=u'tables/{tableId}/styles/{styleId}', request_field='', request_type_name=u'FusiontablesStyleDeleteRequest', response_type_name=u'FusiontablesStyleDeleteResponse', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a specific style.

      Args:
        request: (FusiontablesStyleGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (StyleSetting) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method=u'GET', method_id=u'fusiontables.style.get', ordered_params=[u'tableId', u'styleId'], path_params=[u'styleId', u'tableId'], query_params=[], relative_path=u'tables/{tableId}/styles/{styleId}', request_field='', request_type_name=u'FusiontablesStyleGetRequest', response_type_name=u'StyleSetting', supports_download=False)

    def Insert(self, request, global_params=None):
        """Adds a new style for the table.

      Args:
        request: (StyleSetting) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (StyleSetting) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method=u'POST', method_id=u'fusiontables.style.insert', ordered_params=[u'tableId'], path_params=[u'tableId'], query_params=[], relative_path=u'tables/{tableId}/styles', request_field='<request>', request_type_name=u'StyleSetting', response_type_name=u'StyleSetting', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves a list of styles.

      Args:
        request: (FusiontablesStyleListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (StyleSettingList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method=u'GET', method_id=u'fusiontables.style.list', ordered_params=[u'tableId'], path_params=[u'tableId'], query_params=[u'maxResults', u'pageToken'], relative_path=u'tables/{tableId}/styles', request_field='', request_type_name=u'FusiontablesStyleListRequest', response_type_name=u'StyleSettingList', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates an existing style. This method supports patch semantics.

      Args:
        request: (StyleSetting) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (StyleSetting) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(http_method=u'PATCH', method_id=u'fusiontables.style.patch', ordered_params=[u'tableId', u'styleId'], path_params=[u'styleId', u'tableId'], query_params=[], relative_path=u'tables/{tableId}/styles/{styleId}', request_field='<request>', request_type_name=u'StyleSetting', response_type_name=u'StyleSetting', supports_download=False)

    def Update(self, request, global_params=None):
        """Updates an existing style.

      Args:
        request: (StyleSetting) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (StyleSetting) The response message.
      """
        config = self.GetMethodConfig('Update')
        return self._RunMethod(config, request, global_params=global_params)
    Update.method_config = lambda: base_api.ApiMethodInfo(http_method=u'PUT', method_id=u'fusiontables.style.update', ordered_params=[u'tableId', u'styleId'], path_params=[u'styleId', u'tableId'], query_params=[], relative_path=u'tables/{tableId}/styles/{styleId}', request_field='<request>', request_type_name=u'StyleSetting', response_type_name=u'StyleSetting', supports_download=False)