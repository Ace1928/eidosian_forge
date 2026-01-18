from __future__ import absolute_import, division, print_function
import json
import os
import time
import traceback
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib

        Find the clc server by the UUID returned from the provisioning request.  Retry the request if a 404 is returned.
        :param clc: the clc-sdk instance to use
        :param module: the AnsibleModule object
        :param svr_uuid: UUID of the server
        :param retries: the number of retry attempts to make prior to fail. default is 5
        :param alias: the Account Alias to search
        :return: a clc-sdk.Server instance
        