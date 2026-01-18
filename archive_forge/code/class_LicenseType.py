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
class LicenseType:
    _license_classes = {'import': ImportLicense, 'export': ExportLicense, 'delete': DeleteLicense}

    @staticmethod
    def license_operation(idrac, module):
        """
        Perform a license operation based on the given parameters.

        :param idrac: The IDRAC object.
        :type idrac: IDRAC
        :param module: The Ansible module object.
        :type module: AnsibleModule
        :return: The license class object based on the license type.
        :rtype: LicenseType
        """
        license_type = next((param for param in ['import', 'export', 'delete'] if module.params[param]), None)
        if not license_type:
            module.exit_json(msg=NO_OPERATION_SKIP_MSG, skipped=True)
        license_class = LicenseType._license_classes.get(license_type)
        return license_class(idrac, module)