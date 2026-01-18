from __future__ import (absolute_import, division, print_function)
import json
import os
import base64
from urllib.error import HTTPError, URLError
from ansible_collections.dellemc.openmanage.plugins.module_utils.idrac_redfish import iDRACRedfishAPI, idrac_auth_params
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.compat.version import LooseVersion
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import (
def __import_license_http(self, import_license_url, resource_id):
    """
        Imports a license using HTTP.

        Args:
            module (object): The Ansible module object.
            import_license_url (str): The URL for importing the license.
            resource_id (str): The ID of the resource.

        Returns:
            object: The import status.
        """
    payload = {}
    payload['LicenseName'] = self.module.params.get('share_parameters').get('file_name')
    payload['FQDD'] = resource_id
    payload['ImportOptions'] = 'Force'
    proxy_details = self.get_proxy_details()
    payload.update(proxy_details)
    import_status = self.idrac.invoke_request(import_license_url, 'POST', data=payload)
    return import_status