from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.admin.v1 import admin_v1_messages as messages
class MembersService(base_api.BaseApiService):
    """Service class for the members resource."""
    _NAME = u'members'

    def __init__(self, client):
        super(AdminDirectoryV1.MembersService, self).__init__(client)
        self._upload_configs = {}

    def Delete(self, request, global_params=None):
        """Remove membership.

      Args:
        request: (DirectoryMembersDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (DirectoryMembersDeleteResponse) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method=u'DELETE', method_id=u'directory.members.delete', ordered_params=[u'groupKey', u'memberKey'], path_params=[u'groupKey', u'memberKey'], query_params=[], relative_path=u'groups/{groupKey}/members/{memberKey}', request_field='', request_type_name=u'DirectoryMembersDeleteRequest', response_type_name=u'DirectoryMembersDeleteResponse', supports_download=False)

    def Get(self, request, global_params=None):
        """Retrieve Group Member.

      Args:
        request: (DirectoryMembersGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (Member) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method=u'GET', method_id=u'directory.members.get', ordered_params=[u'groupKey', u'memberKey'], path_params=[u'groupKey', u'memberKey'], query_params=[], relative_path=u'groups/{groupKey}/members/{memberKey}', request_field='', request_type_name=u'DirectoryMembersGetRequest', response_type_name=u'Member', supports_download=False)

    def HasMember(self, request, global_params=None):
        """Checks whether the given user is a member of the group.

      Membership can be direct or nested.

      Args:
        request: (DirectoryMembersHasMemberRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (MembersHasMember) The response message.
      """
        config = self.GetMethodConfig('HasMember')
        return self._RunMethod(config, request, global_params=global_params)
    HasMember.method_config = lambda: base_api.ApiMethodInfo(http_method=u'GET', method_id=u'directory.members.hasMember', ordered_params=[u'groupKey', u'memberKey'], path_params=[u'groupKey', u'memberKey'], query_params=[], relative_path=u'groups/{groupKey}/hasMember/{memberKey}', request_field='', request_type_name=u'DirectoryMembersHasMemberRequest', response_type_name=u'MembersHasMember', supports_download=False)

    def Insert(self, request, global_params=None):
        """Add user to the specified group.

      Args:
        request: (DirectoryMembersInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (Member) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method=u'POST', method_id=u'directory.members.insert', ordered_params=[u'groupKey'], path_params=[u'groupKey'], query_params=[], relative_path=u'groups/{groupKey}/members', request_field=u'member', request_type_name=u'DirectoryMembersInsertRequest', response_type_name=u'Member', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieve all members in a group (paginated).

      Args:
        request: (DirectoryMembersListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (Members) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method=u'GET', method_id=u'directory.members.list', ordered_params=[u'groupKey'], path_params=[u'groupKey'], query_params=[u'includeDerivedMembership', u'maxResults', u'pageToken', u'roles'], relative_path=u'groups/{groupKey}/members', request_field='', request_type_name=u'DirectoryMembersListRequest', response_type_name=u'Members', supports_download=False)

    def Patch(self, request, global_params=None):
        """Update membership of a user in the specified group.

      This method supports patch semantics.

      Args:
        request: (DirectoryMembersPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (Member) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(http_method=u'PATCH', method_id=u'directory.members.patch', ordered_params=[u'groupKey', u'memberKey'], path_params=[u'groupKey', u'memberKey'], query_params=[], relative_path=u'groups/{groupKey}/members/{memberKey}', request_field=u'member', request_type_name=u'DirectoryMembersPatchRequest', response_type_name=u'Member', supports_download=False)

    def Update(self, request, global_params=None):
        """Update membership of a user in the specified group.

      Args:
        request: (DirectoryMembersUpdateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (Member) The response message.
      """
        config = self.GetMethodConfig('Update')
        return self._RunMethod(config, request, global_params=global_params)
    Update.method_config = lambda: base_api.ApiMethodInfo(http_method=u'PUT', method_id=u'directory.members.update', ordered_params=[u'groupKey', u'memberKey'], path_params=[u'groupKey', u'memberKey'], query_params=[], relative_path=u'groups/{groupKey}/members/{memberKey}', request_field=u'member', request_type_name=u'DirectoryMembersUpdateRequest', response_type_name=u'Member', supports_download=False)