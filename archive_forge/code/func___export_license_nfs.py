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
def __export_license_nfs(self, export_license_url):
    """
        Export the license using NFS share type.

        Args:
            module (object): The Ansible module object.
            export_license_url (str): The URL for exporting the license.

        Returns:
            dict: The export status of the license.
        """
    payload = {}
    payload['EntitlementID'] = self.module.params.get('license_id')
    payload['ShareType'] = 'NFS'
    payload['IPAddress'] = self.module.params.get('share_parameters').get('ip_address')
    payload['ShareName'] = self.module.params.get('share_parameters').get('share_name')
    export_status = self.__export_license(payload, export_license_url)
    return export_status