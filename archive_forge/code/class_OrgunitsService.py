from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.admin.v1 import admin_v1_messages as messages
class OrgunitsService(base_api.BaseApiService):
    """Service class for the orgunits resource."""
    _NAME = u'orgunits'

    def __init__(self, client):
        super(AdminDirectoryV1.OrgunitsService, self).__init__(client)
        self._upload_configs = {}

    def Delete(self, request, global_params=None):
        """Remove organizational unit.

      Args:
        request: (DirectoryOrgunitsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (DirectoryOrgunitsDeleteResponse) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method=u'DELETE', method_id=u'directory.orgunits.delete', ordered_params=[u'customerId', u'orgUnitPath'], path_params=[u'customerId', u'orgUnitPath'], query_params=[], relative_path=u'customer/{customerId}/orgunits{/orgUnitPath*}', request_field='', request_type_name=u'DirectoryOrgunitsDeleteRequest', response_type_name=u'DirectoryOrgunitsDeleteResponse', supports_download=False)

    def Get(self, request, global_params=None):
        """Retrieve organizational unit.

      Args:
        request: (DirectoryOrgunitsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (OrgUnit) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method=u'GET', method_id=u'directory.orgunits.get', ordered_params=[u'customerId', u'orgUnitPath'], path_params=[u'customerId', u'orgUnitPath'], query_params=[], relative_path=u'customer/{customerId}/orgunits{/orgUnitPath*}', request_field='', request_type_name=u'DirectoryOrgunitsGetRequest', response_type_name=u'OrgUnit', supports_download=False)

    def Insert(self, request, global_params=None):
        """Add organizational unit.

      Args:
        request: (DirectoryOrgunitsInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (OrgUnit) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method=u'POST', method_id=u'directory.orgunits.insert', ordered_params=[u'customerId'], path_params=[u'customerId'], query_params=[], relative_path=u'customer/{customerId}/orgunits', request_field=u'orgUnit', request_type_name=u'DirectoryOrgunitsInsertRequest', response_type_name=u'OrgUnit', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieve all organizational units.

      Args:
        request: (DirectoryOrgunitsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (OrgUnits) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method=u'GET', method_id=u'directory.orgunits.list', ordered_params=[u'customerId'], path_params=[u'customerId'], query_params=[u'orgUnitPath', u'type'], relative_path=u'customer/{customerId}/orgunits', request_field='', request_type_name=u'DirectoryOrgunitsListRequest', response_type_name=u'OrgUnits', supports_download=False)

    def Patch(self, request, global_params=None):
        """Update organizational unit.

      This method supports patch semantics.

      Args:
        request: (DirectoryOrgunitsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (OrgUnit) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(http_method=u'PATCH', method_id=u'directory.orgunits.patch', ordered_params=[u'customerId', u'orgUnitPath'], path_params=[u'customerId', u'orgUnitPath'], query_params=[], relative_path=u'customer/{customerId}/orgunits{/orgUnitPath*}', request_field=u'orgUnit', request_type_name=u'DirectoryOrgunitsPatchRequest', response_type_name=u'OrgUnit', supports_download=False)

    def Update(self, request, global_params=None):
        """Update organizational unit.

      Args:
        request: (DirectoryOrgunitsUpdateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (OrgUnit) The response message.
      """
        config = self.GetMethodConfig('Update')
        return self._RunMethod(config, request, global_params=global_params)
    Update.method_config = lambda: base_api.ApiMethodInfo(http_method=u'PUT', method_id=u'directory.orgunits.update', ordered_params=[u'customerId', u'orgUnitPath'], path_params=[u'customerId', u'orgUnitPath'], query_params=[], relative_path=u'customer/{customerId}/orgunits{/orgUnitPath*}', request_field=u'orgUnit', request_type_name=u'DirectoryOrgunitsUpdateRequest', response_type_name=u'OrgUnit', supports_download=False)