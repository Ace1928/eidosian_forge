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
class DeleteLicense(License):

    def execute(self):
        """
        Executes the delete operation for a given license ID.

        Args:
            module (object): The Ansible module object.

        Returns:
            object: The response object from the delete operation.
        """
        license_id = self.module.params.get('license_id')
        self.check_license_id(license_id)
        license_url = self.get_license_url()
        delete_license_url = license_url + f'/{license_id}'
        delete_license_response = self.idrac.invoke_request(delete_license_url, 'DELETE')
        status = delete_license_response.status_code
        if status == 204:
            self.module.exit_json(msg=SUCCESS_DELETE_MSG, changed=True)
        else:
            self.module.exit_json(msg=FAILURE_MSG.format(operation='delete', license_id=license_id), failed=True)