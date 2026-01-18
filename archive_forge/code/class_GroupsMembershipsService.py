from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudidentity.v1 import cloudidentity_v1_messages as messages
class GroupsMembershipsService(base_api.BaseApiService):
    """Service class for the groups_memberships resource."""
    _NAME = 'groups_memberships'

    def __init__(self, client):
        super(CloudidentityV1.GroupsMembershipsService, self).__init__(client)
        self._upload_configs = {}

    def CheckTransitiveMembership(self, request, global_params=None):
        """Check a potential member for membership in a group. **Note:** This feature is only available to Google Workspace Enterprise Standard, Enterprise Plus, and Enterprise for Education; and Cloud Identity Premium accounts. If the account of the member is not one of these, a 403 (PERMISSION_DENIED) HTTP status code will be returned. A member has membership to a group as long as there is a single viewable transitive membership between the group and the member. The actor must have view permissions to at least one transitive membership between the member and group.

      Args:
        request: (CloudidentityGroupsMembershipsCheckTransitiveMembershipRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (CheckTransitiveMembershipResponse) The response message.
      """
        config = self.GetMethodConfig('CheckTransitiveMembership')
        return self._RunMethod(config, request, global_params=global_params)
    CheckTransitiveMembership.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/groups/{groupsId}/memberships:checkTransitiveMembership', http_method='GET', method_id='cloudidentity.groups.memberships.checkTransitiveMembership', ordered_params=['parent'], path_params=['parent'], query_params=['query'], relative_path='v1/{+parent}/memberships:checkTransitiveMembership', request_field='', request_type_name='CloudidentityGroupsMembershipsCheckTransitiveMembershipRequest', response_type_name='CheckTransitiveMembershipResponse', supports_download=False)

    def Create(self, request, global_params=None):
        """Creates a `Membership`.

      Args:
        request: (CloudidentityGroupsMembershipsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/groups/{groupsId}/memberships', http_method='POST', method_id='cloudidentity.groups.memberships.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/memberships', request_field='membership', request_type_name='CloudidentityGroupsMembershipsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a `Membership`.

      Args:
        request: (CloudidentityGroupsMembershipsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/groups/{groupsId}/memberships/{membershipsId}', http_method='DELETE', method_id='cloudidentity.groups.memberships.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='CloudidentityGroupsMembershipsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Retrieves a `Membership`.

      Args:
        request: (CloudidentityGroupsMembershipsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Membership) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/groups/{groupsId}/memberships/{membershipsId}', http_method='GET', method_id='cloudidentity.groups.memberships.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='CloudidentityGroupsMembershipsGetRequest', response_type_name='Membership', supports_download=False)

    def GetMembershipGraph(self, request, global_params=None):
        """Get a membership graph of just a member or both a member and a group. **Note:** This feature is only available to Google Workspace Enterprise Standard, Enterprise Plus, and Enterprise for Education; and Cloud Identity Premium accounts. If the account of the member is not one of these, a 403 (PERMISSION_DENIED) HTTP status code will be returned. Given a member, the response will contain all membership paths from the member. Given both a group and a member, the response will contain all membership paths between the group and the member.

      Args:
        request: (CloudidentityGroupsMembershipsGetMembershipGraphRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('GetMembershipGraph')
        return self._RunMethod(config, request, global_params=global_params)
    GetMembershipGraph.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/groups/{groupsId}/memberships:getMembershipGraph', http_method='GET', method_id='cloudidentity.groups.memberships.getMembershipGraph', ordered_params=['parent'], path_params=['parent'], query_params=['query'], relative_path='v1/{+parent}/memberships:getMembershipGraph', request_field='', request_type_name='CloudidentityGroupsMembershipsGetMembershipGraphRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Lists the `Membership`s within a `Group`.

      Args:
        request: (CloudidentityGroupsMembershipsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListMembershipsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/groups/{groupsId}/memberships', http_method='GET', method_id='cloudidentity.groups.memberships.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken', 'view'], relative_path='v1/{+parent}/memberships', request_field='', request_type_name='CloudidentityGroupsMembershipsListRequest', response_type_name='ListMembershipsResponse', supports_download=False)

    def Lookup(self, request, global_params=None):
        """Looks up the [resource name](https://cloud.google.com/apis/design/resource_names) of a `Membership` by its `EntityKey`.

      Args:
        request: (CloudidentityGroupsMembershipsLookupRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (LookupMembershipNameResponse) The response message.
      """
        config = self.GetMethodConfig('Lookup')
        return self._RunMethod(config, request, global_params=global_params)
    Lookup.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/groups/{groupsId}/memberships:lookup', http_method='GET', method_id='cloudidentity.groups.memberships.lookup', ordered_params=['parent'], path_params=['parent'], query_params=['memberKey_id', 'memberKey_namespace'], relative_path='v1/{+parent}/memberships:lookup', request_field='', request_type_name='CloudidentityGroupsMembershipsLookupRequest', response_type_name='LookupMembershipNameResponse', supports_download=False)

    def ModifyMembershipRoles(self, request, global_params=None):
        """Modifies the `MembershipRole`s of a `Membership`.

      Args:
        request: (CloudidentityGroupsMembershipsModifyMembershipRolesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ModifyMembershipRolesResponse) The response message.
      """
        config = self.GetMethodConfig('ModifyMembershipRoles')
        return self._RunMethod(config, request, global_params=global_params)
    ModifyMembershipRoles.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/groups/{groupsId}/memberships/{membershipsId}:modifyMembershipRoles', http_method='POST', method_id='cloudidentity.groups.memberships.modifyMembershipRoles', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:modifyMembershipRoles', request_field='modifyMembershipRolesRequest', request_type_name='CloudidentityGroupsMembershipsModifyMembershipRolesRequest', response_type_name='ModifyMembershipRolesResponse', supports_download=False)

    def SearchDirectGroups(self, request, global_params=None):
        """Searches direct groups of a member.

      Args:
        request: (CloudidentityGroupsMembershipsSearchDirectGroupsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SearchDirectGroupsResponse) The response message.
      """
        config = self.GetMethodConfig('SearchDirectGroups')
        return self._RunMethod(config, request, global_params=global_params)
    SearchDirectGroups.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/groups/{groupsId}/memberships:searchDirectGroups', http_method='GET', method_id='cloudidentity.groups.memberships.searchDirectGroups', ordered_params=['parent'], path_params=['parent'], query_params=['orderBy', 'pageSize', 'pageToken', 'query'], relative_path='v1/{+parent}/memberships:searchDirectGroups', request_field='', request_type_name='CloudidentityGroupsMembershipsSearchDirectGroupsRequest', response_type_name='SearchDirectGroupsResponse', supports_download=False)

    def SearchTransitiveGroups(self, request, global_params=None):
        """Search transitive groups of a member. **Note:** This feature is only available to Google Workspace Enterprise Standard, Enterprise Plus, and Enterprise for Education; and Cloud Identity Premium accounts. If the account of the member is not one of these, a 403 (PERMISSION_DENIED) HTTP status code will be returned. A transitive group is any group that has a direct or indirect membership to the member. Actor must have view permissions all transitive groups.

      Args:
        request: (CloudidentityGroupsMembershipsSearchTransitiveGroupsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SearchTransitiveGroupsResponse) The response message.
      """
        config = self.GetMethodConfig('SearchTransitiveGroups')
        return self._RunMethod(config, request, global_params=global_params)
    SearchTransitiveGroups.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/groups/{groupsId}/memberships:searchTransitiveGroups', http_method='GET', method_id='cloudidentity.groups.memberships.searchTransitiveGroups', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken', 'query'], relative_path='v1/{+parent}/memberships:searchTransitiveGroups', request_field='', request_type_name='CloudidentityGroupsMembershipsSearchTransitiveGroupsRequest', response_type_name='SearchTransitiveGroupsResponse', supports_download=False)

    def SearchTransitiveMemberships(self, request, global_params=None):
        """Search transitive memberships of a group. **Note:** This feature is only available to Google Workspace Enterprise Standard, Enterprise Plus, and Enterprise for Education; and Cloud Identity Premium accounts. If the account of the group is not one of these, a 403 (PERMISSION_DENIED) HTTP status code will be returned. A transitive membership is any direct or indirect membership of a group. Actor must have view permissions to all transitive memberships.

      Args:
        request: (CloudidentityGroupsMembershipsSearchTransitiveMembershipsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SearchTransitiveMembershipsResponse) The response message.
      """
        config = self.GetMethodConfig('SearchTransitiveMemberships')
        return self._RunMethod(config, request, global_params=global_params)
    SearchTransitiveMemberships.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/groups/{groupsId}/memberships:searchTransitiveMemberships', http_method='GET', method_id='cloudidentity.groups.memberships.searchTransitiveMemberships', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/memberships:searchTransitiveMemberships', request_field='', request_type_name='CloudidentityGroupsMembershipsSearchTransitiveMembershipsRequest', response_type_name='SearchTransitiveMembershipsResponse', supports_download=False)