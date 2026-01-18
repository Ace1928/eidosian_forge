from __future__ import absolute_import
import os
import platform
import sys
from apitools.base.py import base_api
import gslib.third_party.kms_apitools.cloudkms_v1_messages as messages
import gslib
from gslib.metrics import MetricsCollector
from gslib.utils import system_util
class ProjectsLocationsKeyRingsCryptoKeysService(base_api.BaseApiService):
    """Service class for the projects_locations_keyRings_cryptoKeys resource."""
    _NAME = u'projects_locations_keyRings_cryptoKeys'

    def __init__(self, client):
        super(CloudkmsV1.ProjectsLocationsKeyRingsCryptoKeysService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Create a new CryptoKey within a KeyRing.

CryptoKey.purpose is required.

      Args:
        request: (CloudkmsProjectsLocationsKeyRingsCryptoKeysCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (CryptoKey) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path=u'v1/projects/{projectsId}/locations/{locationsId}/keyRings/{keyRingsId}/cryptoKeys', http_method=u'POST', method_id=u'cloudkms.projects.locations.keyRings.cryptoKeys.create', ordered_params=[u'parent'], path_params=[u'parent'], query_params=[u'cryptoKeyId'], relative_path=u'v1/{+parent}/cryptoKeys', request_field=u'cryptoKey', request_type_name=u'CloudkmsProjectsLocationsKeyRingsCryptoKeysCreateRequest', response_type_name=u'CryptoKey', supports_download=False)

    def Decrypt(self, request, global_params=None):
        """Decrypts data that was protected by Encrypt.

      Args:
        request: (CloudkmsProjectsLocationsKeyRingsCryptoKeysDecryptRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (DecryptResponse) The response message.
      """
        config = self.GetMethodConfig('Decrypt')
        return self._RunMethod(config, request, global_params=global_params)
    Decrypt.method_config = lambda: base_api.ApiMethodInfo(flat_path=u'v1/projects/{projectsId}/locations/{locationsId}/keyRings/{keyRingsId}/cryptoKeys/{cryptoKeysId}:decrypt', http_method=u'POST', method_id=u'cloudkms.projects.locations.keyRings.cryptoKeys.decrypt', ordered_params=[u'name'], path_params=[u'name'], query_params=[], relative_path=u'v1/{+name}:decrypt', request_field=u'decryptRequest', request_type_name=u'CloudkmsProjectsLocationsKeyRingsCryptoKeysDecryptRequest', response_type_name=u'DecryptResponse', supports_download=False)

    def Encrypt(self, request, global_params=None):
        """Encrypts data, so that it can only be recovered by a call to Decrypt.

      Args:
        request: (CloudkmsProjectsLocationsKeyRingsCryptoKeysEncryptRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (EncryptResponse) The response message.
      """
        config = self.GetMethodConfig('Encrypt')
        return self._RunMethod(config, request, global_params=global_params)
    Encrypt.method_config = lambda: base_api.ApiMethodInfo(flat_path=u'v1/projects/{projectsId}/locations/{locationsId}/keyRings/{keyRingsId}/cryptoKeys/{cryptoKeysId}:encrypt', http_method=u'POST', method_id=u'cloudkms.projects.locations.keyRings.cryptoKeys.encrypt', ordered_params=[u'name'], path_params=[u'name'], query_params=[], relative_path=u'v1/{+name}:encrypt', request_field=u'encryptRequest', request_type_name=u'CloudkmsProjectsLocationsKeyRingsCryptoKeysEncryptRequest', response_type_name=u'EncryptResponse', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns metadata for a given CryptoKey, as well as its.
primary CryptoKeyVersion.

      Args:
        request: (CloudkmsProjectsLocationsKeyRingsCryptoKeysGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (CryptoKey) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path=u'v1/projects/{projectsId}/locations/{locationsId}/keyRings/{keyRingsId}/cryptoKeys/{cryptoKeysId}', http_method=u'GET', method_id=u'cloudkms.projects.locations.keyRings.cryptoKeys.get', ordered_params=[u'name'], path_params=[u'name'], query_params=[], relative_path=u'v1/{+name}', request_field='', request_type_name=u'CloudkmsProjectsLocationsKeyRingsCryptoKeysGetRequest', response_type_name=u'CryptoKey', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource.
Returns an empty policy if the resource exists and does not have a policy
set.

      Args:
        request: (CloudkmsProjectsLocationsKeyRingsCryptoKeysGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path=u'v1/projects/{projectsId}/locations/{locationsId}/keyRings/{keyRingsId}/cryptoKeys/{cryptoKeysId}:getIamPolicy', http_method=u'GET', method_id=u'cloudkms.projects.locations.keyRings.cryptoKeys.getIamPolicy', ordered_params=[u'resource'], path_params=[u'resource'], query_params=[], relative_path=u'v1/{+resource}:getIamPolicy', request_field='', request_type_name=u'CloudkmsProjectsLocationsKeyRingsCryptoKeysGetIamPolicyRequest', response_type_name=u'Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists CryptoKeys.

      Args:
        request: (CloudkmsProjectsLocationsKeyRingsCryptoKeysListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListCryptoKeysResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path=u'v1/projects/{projectsId}/locations/{locationsId}/keyRings/{keyRingsId}/cryptoKeys', http_method=u'GET', method_id=u'cloudkms.projects.locations.keyRings.cryptoKeys.list', ordered_params=[u'parent'], path_params=[u'parent'], query_params=[u'pageSize', u'pageToken'], relative_path=u'v1/{+parent}/cryptoKeys', request_field='', request_type_name=u'CloudkmsProjectsLocationsKeyRingsCryptoKeysListRequest', response_type_name=u'ListCryptoKeysResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Update a CryptoKey.

      Args:
        request: (CloudkmsProjectsLocationsKeyRingsCryptoKeysPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (CryptoKey) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path=u'v1/projects/{projectsId}/locations/{locationsId}/keyRings/{keyRingsId}/cryptoKeys/{cryptoKeysId}', http_method=u'PATCH', method_id=u'cloudkms.projects.locations.keyRings.cryptoKeys.patch', ordered_params=[u'name'], path_params=[u'name'], query_params=[u'updateMask'], relative_path=u'v1/{+name}', request_field=u'cryptoKey', request_type_name=u'CloudkmsProjectsLocationsKeyRingsCryptoKeysPatchRequest', response_type_name=u'CryptoKey', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any.
existing policy.

      Args:
        request: (CloudkmsProjectsLocationsKeyRingsCryptoKeysSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path=u'v1/projects/{projectsId}/locations/{locationsId}/keyRings/{keyRingsId}/cryptoKeys/{cryptoKeysId}:setIamPolicy', http_method=u'POST', method_id=u'cloudkms.projects.locations.keyRings.cryptoKeys.setIamPolicy', ordered_params=[u'resource'], path_params=[u'resource'], query_params=[], relative_path=u'v1/{+resource}:setIamPolicy', request_field=u'setIamPolicyRequest', request_type_name=u'CloudkmsProjectsLocationsKeyRingsCryptoKeysSetIamPolicyRequest', response_type_name=u'Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource.
If the resource does not exist, this will return an empty set of
permissions, not a NOT_FOUND error.

Note: This operation is designed to be used for building permission-aware
UIs and command-line tools, not for authorization checking. This operation
may "fail open" without warning.

      Args:
        request: (CloudkmsProjectsLocationsKeyRingsCryptoKeysTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path=u'v1/projects/{projectsId}/locations/{locationsId}/keyRings/{keyRingsId}/cryptoKeys/{cryptoKeysId}:testIamPermissions', http_method=u'POST', method_id=u'cloudkms.projects.locations.keyRings.cryptoKeys.testIamPermissions', ordered_params=[u'resource'], path_params=[u'resource'], query_params=[], relative_path=u'v1/{+resource}:testIamPermissions', request_field=u'testIamPermissionsRequest', request_type_name=u'CloudkmsProjectsLocationsKeyRingsCryptoKeysTestIamPermissionsRequest', response_type_name=u'TestIamPermissionsResponse', supports_download=False)

    def UpdatePrimaryVersion(self, request, global_params=None):
        """Update the version of a CryptoKey that will be used in Encrypt.

      Args:
        request: (CloudkmsProjectsLocationsKeyRingsCryptoKeysUpdatePrimaryVersionRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (CryptoKey) The response message.
      """
        config = self.GetMethodConfig('UpdatePrimaryVersion')
        return self._RunMethod(config, request, global_params=global_params)
    UpdatePrimaryVersion.method_config = lambda: base_api.ApiMethodInfo(flat_path=u'v1/projects/{projectsId}/locations/{locationsId}/keyRings/{keyRingsId}/cryptoKeys/{cryptoKeysId}:updatePrimaryVersion', http_method=u'POST', method_id=u'cloudkms.projects.locations.keyRings.cryptoKeys.updatePrimaryVersion', ordered_params=[u'name'], path_params=[u'name'], query_params=[], relative_path=u'v1/{+name}:updatePrimaryVersion', request_field=u'updateCryptoKeyPrimaryVersionRequest', request_type_name=u'CloudkmsProjectsLocationsKeyRingsCryptoKeysUpdatePrimaryVersionRequest', response_type_name=u'CryptoKey', supports_download=False)