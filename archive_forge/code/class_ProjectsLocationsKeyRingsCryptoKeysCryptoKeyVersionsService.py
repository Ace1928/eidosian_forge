from __future__ import absolute_import
import os
import platform
import sys
from apitools.base.py import base_api
import gslib.third_party.kms_apitools.cloudkms_v1_messages as messages
import gslib
from gslib.metrics import MetricsCollector
from gslib.utils import system_util
class ProjectsLocationsKeyRingsCryptoKeysCryptoKeyVersionsService(base_api.BaseApiService):
    """Service class for the projects_locations_keyRings_cryptoKeys_cryptoKeyVersions resource."""
    _NAME = u'projects_locations_keyRings_cryptoKeys_cryptoKeyVersions'

    def __init__(self, client):
        super(CloudkmsV1.ProjectsLocationsKeyRingsCryptoKeysCryptoKeyVersionsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Create a new CryptoKeyVersion in a CryptoKey.

The server will assign the next sequential id. If unset,
state will be set to
ENABLED.

      Args:
        request: (CloudkmsProjectsLocationsKeyRingsCryptoKeysCryptoKeyVersionsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (CryptoKeyVersion) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path=u'v1/projects/{projectsId}/locations/{locationsId}/keyRings/{keyRingsId}/cryptoKeys/{cryptoKeysId}/cryptoKeyVersions', http_method=u'POST', method_id=u'cloudkms.projects.locations.keyRings.cryptoKeys.cryptoKeyVersions.create', ordered_params=[u'parent'], path_params=[u'parent'], query_params=[], relative_path=u'v1/{+parent}/cryptoKeyVersions', request_field=u'cryptoKeyVersion', request_type_name=u'CloudkmsProjectsLocationsKeyRingsCryptoKeysCryptoKeyVersionsCreateRequest', response_type_name=u'CryptoKeyVersion', supports_download=False)

    def Destroy(self, request, global_params=None):
        """Schedule a CryptoKeyVersion for destruction.

Upon calling this method, CryptoKeyVersion.state will be set to
DESTROY_SCHEDULED
and destroy_time will be set to a time 24
hours in the future, at which point the state
will be changed to
DESTROYED, and the key
material will be irrevocably destroyed.

Before the destroy_time is reached,
RestoreCryptoKeyVersion may be called to reverse the process.

      Args:
        request: (CloudkmsProjectsLocationsKeyRingsCryptoKeysCryptoKeyVersionsDestroyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (CryptoKeyVersion) The response message.
      """
        config = self.GetMethodConfig('Destroy')
        return self._RunMethod(config, request, global_params=global_params)
    Destroy.method_config = lambda: base_api.ApiMethodInfo(flat_path=u'v1/projects/{projectsId}/locations/{locationsId}/keyRings/{keyRingsId}/cryptoKeys/{cryptoKeysId}/cryptoKeyVersions/{cryptoKeyVersionsId}:destroy', http_method=u'POST', method_id=u'cloudkms.projects.locations.keyRings.cryptoKeys.cryptoKeyVersions.destroy', ordered_params=[u'name'], path_params=[u'name'], query_params=[], relative_path=u'v1/{+name}:destroy', request_field=u'destroyCryptoKeyVersionRequest', request_type_name=u'CloudkmsProjectsLocationsKeyRingsCryptoKeysCryptoKeyVersionsDestroyRequest', response_type_name=u'CryptoKeyVersion', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns metadata for a given CryptoKeyVersion.

      Args:
        request: (CloudkmsProjectsLocationsKeyRingsCryptoKeysCryptoKeyVersionsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (CryptoKeyVersion) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path=u'v1/projects/{projectsId}/locations/{locationsId}/keyRings/{keyRingsId}/cryptoKeys/{cryptoKeysId}/cryptoKeyVersions/{cryptoKeyVersionsId}', http_method=u'GET', method_id=u'cloudkms.projects.locations.keyRings.cryptoKeys.cryptoKeyVersions.get', ordered_params=[u'name'], path_params=[u'name'], query_params=[], relative_path=u'v1/{+name}', request_field='', request_type_name=u'CloudkmsProjectsLocationsKeyRingsCryptoKeysCryptoKeyVersionsGetRequest', response_type_name=u'CryptoKeyVersion', supports_download=False)

    def List(self, request, global_params=None):
        """Lists CryptoKeyVersions.

      Args:
        request: (CloudkmsProjectsLocationsKeyRingsCryptoKeysCryptoKeyVersionsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListCryptoKeyVersionsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path=u'v1/projects/{projectsId}/locations/{locationsId}/keyRings/{keyRingsId}/cryptoKeys/{cryptoKeysId}/cryptoKeyVersions', http_method=u'GET', method_id=u'cloudkms.projects.locations.keyRings.cryptoKeys.cryptoKeyVersions.list', ordered_params=[u'parent'], path_params=[u'parent'], query_params=[u'pageSize', u'pageToken'], relative_path=u'v1/{+parent}/cryptoKeyVersions', request_field='', request_type_name=u'CloudkmsProjectsLocationsKeyRingsCryptoKeysCryptoKeyVersionsListRequest', response_type_name=u'ListCryptoKeyVersionsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Update a CryptoKeyVersion's metadata.

state may be changed between
ENABLED and
DISABLED using this
method. See DestroyCryptoKeyVersion and RestoreCryptoKeyVersion to
move between other states.

      Args:
        request: (CloudkmsProjectsLocationsKeyRingsCryptoKeysCryptoKeyVersionsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (CryptoKeyVersion) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path=u'v1/projects/{projectsId}/locations/{locationsId}/keyRings/{keyRingsId}/cryptoKeys/{cryptoKeysId}/cryptoKeyVersions/{cryptoKeyVersionsId}', http_method=u'PATCH', method_id=u'cloudkms.projects.locations.keyRings.cryptoKeys.cryptoKeyVersions.patch', ordered_params=[u'name'], path_params=[u'name'], query_params=[u'updateMask'], relative_path=u'v1/{+name}', request_field=u'cryptoKeyVersion', request_type_name=u'CloudkmsProjectsLocationsKeyRingsCryptoKeysCryptoKeyVersionsPatchRequest', response_type_name=u'CryptoKeyVersion', supports_download=False)

    def Restore(self, request, global_params=None):
        """Restore a CryptoKeyVersion in the.
DESTROY_SCHEDULED,
state.

Upon restoration of the CryptoKeyVersion, state
will be set to DISABLED,
and destroy_time will be cleared.

      Args:
        request: (CloudkmsProjectsLocationsKeyRingsCryptoKeysCryptoKeyVersionsRestoreRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (CryptoKeyVersion) The response message.
      """
        config = self.GetMethodConfig('Restore')
        return self._RunMethod(config, request, global_params=global_params)
    Restore.method_config = lambda: base_api.ApiMethodInfo(flat_path=u'v1/projects/{projectsId}/locations/{locationsId}/keyRings/{keyRingsId}/cryptoKeys/{cryptoKeysId}/cryptoKeyVersions/{cryptoKeyVersionsId}:restore', http_method=u'POST', method_id=u'cloudkms.projects.locations.keyRings.cryptoKeys.cryptoKeyVersions.restore', ordered_params=[u'name'], path_params=[u'name'], query_params=[], relative_path=u'v1/{+name}:restore', request_field=u'restoreCryptoKeyVersionRequest', request_type_name=u'CloudkmsProjectsLocationsKeyRingsCryptoKeysCryptoKeyVersionsRestoreRequest', response_type_name=u'CryptoKeyVersion', supports_download=False)