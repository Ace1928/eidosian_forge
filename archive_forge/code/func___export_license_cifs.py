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
def __export_license_cifs(self, export_license_url):
    """
        Export the license using CIFS share type.

        Args:
            module (object): The Ansible module object.
            export_license_url (str): The URL for exporting the license.

        Returns:
            str: The export status.
        """
    payload = {}
    payload['EntitlementID'] = self.module.params.get('license_id')
    payload['ShareType'] = 'CIFS'
    if self.module.params.get('share_parameters').get('workgroup'):
        payload['Workgroup'] = self.module.params.get('share_parameters').get('workgroup')
    share_details = self.get_share_details()
    payload.update(share_details)
    export_status = self.__export_license(payload, export_license_url)
    return export_status