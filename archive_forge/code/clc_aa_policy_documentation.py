from __future__ import absolute_import, division, print_function
import os
import traceback
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib

        Ensures that a policy is present
        :param p: dictionary of a policy name
        :return: tuple of if an addition occurred and the name of the policy that was added
        