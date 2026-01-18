from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.admin.v1 import admin_v1_messages as messages
class UsersPhotosService(base_api.BaseApiService):
    """Service class for the users_photos resource."""
    _NAME = u'users_photos'

    def __init__(self, client):
        super(AdminDirectoryV1.UsersPhotosService, self).__init__(client)
        self._upload_configs = {}

    def Delete(self, request, global_params=None):
        """Remove photos for the user.

      Args:
        request: (DirectoryUsersPhotosDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (DirectoryUsersPhotosDeleteResponse) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method=u'DELETE', method_id=u'directory.users.photos.delete', ordered_params=[u'userKey'], path_params=[u'userKey'], query_params=[], relative_path=u'users/{userKey}/photos/thumbnail', request_field='', request_type_name=u'DirectoryUsersPhotosDeleteRequest', response_type_name=u'DirectoryUsersPhotosDeleteResponse', supports_download=False)

    def Get(self, request, global_params=None):
        """Retrieve photo of a user.

      Args:
        request: (DirectoryUsersPhotosGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (UserPhoto) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method=u'GET', method_id=u'directory.users.photos.get', ordered_params=[u'userKey'], path_params=[u'userKey'], query_params=[], relative_path=u'users/{userKey}/photos/thumbnail', request_field='', request_type_name=u'DirectoryUsersPhotosGetRequest', response_type_name=u'UserPhoto', supports_download=False)

    def Patch(self, request, global_params=None):
        """Add a photo for the user.

      This method supports patch semantics.

      Args:
        request: (DirectoryUsersPhotosPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (UserPhoto) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(http_method=u'PATCH', method_id=u'directory.users.photos.patch', ordered_params=[u'userKey'], path_params=[u'userKey'], query_params=[], relative_path=u'users/{userKey}/photos/thumbnail', request_field=u'userPhoto', request_type_name=u'DirectoryUsersPhotosPatchRequest', response_type_name=u'UserPhoto', supports_download=False)

    def Update(self, request, global_params=None):
        """Add a photo for the user.

      Args:
        request: (DirectoryUsersPhotosUpdateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (UserPhoto) The response message.
      """
        config = self.GetMethodConfig('Update')
        return self._RunMethod(config, request, global_params=global_params)
    Update.method_config = lambda: base_api.ApiMethodInfo(http_method=u'PUT', method_id=u'directory.users.photos.update', ordered_params=[u'userKey'], path_params=[u'userKey'], query_params=[], relative_path=u'users/{userKey}/photos/thumbnail', request_field=u'userPhoto', request_type_name=u'DirectoryUsersPhotosUpdateRequest', response_type_name=u'UserPhoto', supports_download=False)