from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.admin.v1 import admin_v1_messages as messages
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