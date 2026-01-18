from __future__ import absolute_import, division, print_function
import json
import os
import time
import traceback
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
@staticmethod
def _find_description(module):
    """
        Set the description module param to name if description is blank
        :param module: the module to validate
        :return: string description
        """
    description = module.params.get('description')
    if not description:
        description = module.params.get('name')
    return description