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
def __get_export_license_url(self):
    """
        Get the export license URL.

        :param module: The module object.
        :type module: object
        :return: The export license URL.
        :rtype: str
        """
    uri, error_msg = validate_and_get_first_resource_id_uri(self.module, self.idrac, MANAGERS_URI)
    if error_msg:
        self.module.exit_json(msg=error_msg, failed=True)
    resp = get_dynamic_uri(self.idrac, uri)
    url = resp.get('Links', {}).get(OEM, {}).get(MANUFACTURER, {}).get(LICENSE_MANAGEMENT_SERVICE, {}).get(ODATA, {})
    action_resp = get_dynamic_uri(self.idrac, url)
    license_service = EXPORT_LOCAL if self.module.params.get('share_parameters').get('share_type') == 'local' else EXPORT_NETWORK_SHARE
    export_url = action_resp.get(ACTIONS, {}).get(license_service, {}).get('target', {})
    return export_url