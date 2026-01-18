from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.essentialcontacts.v1 import essentialcontacts_v1_messages as messages
class FoldersContactsService(base_api.BaseApiService):
    """Service class for the folders_contacts resource."""
    _NAME = 'folders_contacts'

    def __init__(self, client):
        super(EssentialcontactsV1.FoldersContactsService, self).__init__(client)
        self._upload_configs = {}

    def Compute(self, request, global_params=None):
        """Lists all contacts for the resource that are subscribed to the specified notification categories, including contacts inherited from any parent resources.

      Args:
        request: (EssentialcontactsFoldersContactsComputeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudEssentialcontactsV1ComputeContactsResponse) The response message.
      """
        config = self.GetMethodConfig('Compute')
        return self._RunMethod(config, request, global_params=global_params)
    Compute.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/folders/{foldersId}/contacts:compute', http_method='GET', method_id='essentialcontacts.folders.contacts.compute', ordered_params=['parent'], path_params=['parent'], query_params=['notificationCategories', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/contacts:compute', request_field='', request_type_name='EssentialcontactsFoldersContactsComputeRequest', response_type_name='GoogleCloudEssentialcontactsV1ComputeContactsResponse', supports_download=False)

    def Create(self, request, global_params=None):
        """Adds a new contact for a resource.

      Args:
        request: (EssentialcontactsFoldersContactsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudEssentialcontactsV1Contact) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/folders/{foldersId}/contacts', http_method='POST', method_id='essentialcontacts.folders.contacts.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/contacts', request_field='googleCloudEssentialcontactsV1Contact', request_type_name='EssentialcontactsFoldersContactsCreateRequest', response_type_name='GoogleCloudEssentialcontactsV1Contact', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a contact.

      Args:
        request: (EssentialcontactsFoldersContactsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleProtobufEmpty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/folders/{foldersId}/contacts/{contactsId}', http_method='DELETE', method_id='essentialcontacts.folders.contacts.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='EssentialcontactsFoldersContactsDeleteRequest', response_type_name='GoogleProtobufEmpty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a single contact.

      Args:
        request: (EssentialcontactsFoldersContactsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudEssentialcontactsV1Contact) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/folders/{foldersId}/contacts/{contactsId}', http_method='GET', method_id='essentialcontacts.folders.contacts.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='EssentialcontactsFoldersContactsGetRequest', response_type_name='GoogleCloudEssentialcontactsV1Contact', supports_download=False)

    def List(self, request, global_params=None):
        """Lists the contacts that have been set on a resource.

      Args:
        request: (EssentialcontactsFoldersContactsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudEssentialcontactsV1ListContactsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/folders/{foldersId}/contacts', http_method='GET', method_id='essentialcontacts.folders.contacts.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/contacts', request_field='', request_type_name='EssentialcontactsFoldersContactsListRequest', response_type_name='GoogleCloudEssentialcontactsV1ListContactsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a contact. Note: A contact's email address cannot be changed.

      Args:
        request: (EssentialcontactsFoldersContactsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudEssentialcontactsV1Contact) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/folders/{foldersId}/contacts/{contactsId}', http_method='PATCH', method_id='essentialcontacts.folders.contacts.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='googleCloudEssentialcontactsV1Contact', request_type_name='EssentialcontactsFoldersContactsPatchRequest', response_type_name='GoogleCloudEssentialcontactsV1Contact', supports_download=False)

    def SendTestMessage(self, request, global_params=None):
        """Allows a contact admin to send a test message to contact to verify that it has been configured correctly.

      Args:
        request: (EssentialcontactsFoldersContactsSendTestMessageRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleProtobufEmpty) The response message.
      """
        config = self.GetMethodConfig('SendTestMessage')
        return self._RunMethod(config, request, global_params=global_params)
    SendTestMessage.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/folders/{foldersId}/contacts:sendTestMessage', http_method='POST', method_id='essentialcontacts.folders.contacts.sendTestMessage', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}/contacts:sendTestMessage', request_field='googleCloudEssentialcontactsV1SendTestMessageRequest', request_type_name='EssentialcontactsFoldersContactsSendTestMessageRequest', response_type_name='GoogleProtobufEmpty', supports_download=False)