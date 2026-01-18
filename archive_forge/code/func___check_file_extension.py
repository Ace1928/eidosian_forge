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
def __check_file_extension(self):
    """
        Check if the file extension of the given file name is valid.

        :param module: The Ansible module object.
        :type module: AnsibleModule

        :return: None
        """
    share_type = self.module.params.get('share_parameters').get('share_type')
    file_name = self.module.params.get('share_parameters').get('file_name')
    valid_extensions = {'.txt', '.xml'} if share_type == 'local' else {'.xml'}
    file_extension = any((file_name.lower().endswith(ext) for ext in valid_extensions))
    if not file_extension:
        self.module.exit_json(msg=INVALID_FILE_MSG, failed=True)