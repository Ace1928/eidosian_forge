from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.admin.v1 import admin_v1_messages as messages
class UsersAliasesService(base_api.BaseApiService):
    """Service class for the users_aliases resource."""
    _NAME = u'users_aliases'

    def __init__(self, client):
        super(AdminDirectoryV1.UsersAliasesService, self).__init__(client)
        self._upload_configs = {}

    def Delete(self, request, global_params=None):
        """Remove a alias for the user.

      Args:
        request: (DirectoryUsersAliasesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (DirectoryUsersAliasesDeleteResponse) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method=u'DELETE', method_id=u'directory.users.aliases.delete', ordered_params=[u'userKey', u'alias'], path_params=[u'alias', u'userKey'], query_params=[], relative_path=u'users/{userKey}/aliases/{alias}', request_field='', request_type_name=u'DirectoryUsersAliasesDeleteRequest', response_type_name=u'DirectoryUsersAliasesDeleteResponse', supports_download=False)

    def Insert(self, request, global_params=None):
        """Add a alias for the user.

      Args:
        request: (DirectoryUsersAliasesInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (Alias) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method=u'POST', method_id=u'directory.users.aliases.insert', ordered_params=[u'userKey'], path_params=[u'userKey'], query_params=[], relative_path=u'users/{userKey}/aliases', request_field=u'alias', request_type_name=u'DirectoryUsersAliasesInsertRequest', response_type_name=u'Alias', supports_download=False)

    def List(self, request, global_params=None):
        """List all aliases for a user.

      Args:
        request: (DirectoryUsersAliasesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (Aliases) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method=u'GET', method_id=u'directory.users.aliases.list', ordered_params=[u'userKey'], path_params=[u'userKey'], query_params=[u'event'], relative_path=u'users/{userKey}/aliases', request_field='', request_type_name=u'DirectoryUsersAliasesListRequest', response_type_name=u'Aliases', supports_download=False)

    def Watch(self, request, global_params=None):
        """Watch for changes in user aliases list.

      Args:
        request: (DirectoryUsersAliasesWatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (Channel) The response message.
      """
        config = self.GetMethodConfig('Watch')
        return self._RunMethod(config, request, global_params=global_params)
    Watch.method_config = lambda: base_api.ApiMethodInfo(http_method=u'POST', method_id=u'directory.users.aliases.watch', ordered_params=[u'userKey'], path_params=[u'userKey'], query_params=[u'event'], relative_path=u'users/{userKey}/aliases/watch', request_field=u'channel', request_type_name=u'DirectoryUsersAliasesWatchRequest', response_type_name=u'Channel', supports_download=False)