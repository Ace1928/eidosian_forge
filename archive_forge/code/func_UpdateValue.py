from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.resourcesettings.v1alpha1 import resourcesettings_v1alpha1_messages as messages
def UpdateValue(self, request, global_params=None):
    """Updates a setting value. Returns a `google.rpc.Status` with `google.rpc.Code.NOT_FOUND` if the setting or the setting value does not exist. Returns a `google.rpc.Status` with `google.rpc.Code.FAILED_PRECONDITION` if the setting is flagged as read only. Returns a `google.rpc.Status` with `google.rpc.Code.ABORTED` if the etag supplied in the request does not match the persisted etag of the setting value. Note: the supplied setting value will perform a full overwrite of all fields.

      Args:
        request: (ResourcesettingsProjectsSettingsUpdateValueRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudResourcesettingsV1alpha1SettingValue) The response message.
      """
    config = self.GetMethodConfig('UpdateValue')
    return self._RunMethod(config, request, global_params=global_params)