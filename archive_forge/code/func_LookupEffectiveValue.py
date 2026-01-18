from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.resourcesettings.v1alpha1 import resourcesettings_v1alpha1_messages as messages
def LookupEffectiveValue(self, request, global_params=None):
    """Computes the effective setting value of a setting at the Cloud resource `parent`. The effective setting value is the calculated setting value at a Cloud resource and evaluates to one of the following options in the given order (the next option is used if the previous one does not exist): 1. the setting value on the given resource 2. the setting value on the given resource's nearest ancestor 3. the setting's default value 4. an empty setting value, defined as a `SettingValue` with all fields unset Returns a `google.rpc.Status` with `google.rpc.Code.NOT_FOUND` if the setting does not exist.

      Args:
        request: (ResourcesettingsProjectsSettingsLookupEffectiveValueRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudResourcesettingsV1alpha1SettingValue) The response message.
      """
    config = self.GetMethodConfig('LookupEffectiveValue')
    return self._RunMethod(config, request, global_params=global_params)