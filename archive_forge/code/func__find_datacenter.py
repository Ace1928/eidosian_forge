from __future__ import absolute_import, division, print_function
import json
import os
import time
import traceback
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
@staticmethod
def _find_datacenter(clc, module):
    """
        Find the datacenter by calling the CLC API.
        :param clc: clc-sdk instance to use
        :param module: module to validate
        :return: clc-sdk.Datacenter instance
        """
    location = module.params.get('location')
    try:
        if not location:
            account = clc.v2.Account()
            location = account.data.get('primaryDataCenter')
        data_center = clc.v2.Datacenter(location)
        return data_center
    except CLCException:
        module.fail_json(msg='Unable to find location: {0}'.format(location))