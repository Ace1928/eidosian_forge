from __future__ import absolute_import, division, print_function
import json
import os
import time
import traceback
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
@staticmethod
def _find_alias(clc, module):
    """
        Find or Validate the Account Alias by calling the CLC API
        :param clc: clc-sdk instance to use
        :param module: module to validate
        :return: clc-sdk.Account instance
        """
    alias = module.params.get('alias')
    if not alias:
        try:
            alias = clc.v2.Account.GetAlias()
        except CLCException as ex:
            module.fail_json(msg='Unable to find account alias. {0}'.format(ex.message))
    return alias