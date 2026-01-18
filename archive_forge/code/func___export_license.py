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
def __export_license(self, payload, export_license_url):
    """
        Export the license to a file.

        Args:
            module (object): The Ansible module object.
            payload (dict): The payload containing the license information.
            export_license_url (str): The URL for exporting the license.

        Returns:
            dict: The license status after exporting.
        """
    license_name = self.module.params.get('share_parameters').get('file_name')
    if license_name:
        license_file_name = f'{license_name}_iDRAC_license.xml'
    else:
        license_file_name = f'{self.module.params['license_id']}_iDRAC_license.xml'
    payload['FileName'] = license_file_name
    license_status = self.idrac.invoke_request(export_license_url, 'POST', data=payload)
    return license_status