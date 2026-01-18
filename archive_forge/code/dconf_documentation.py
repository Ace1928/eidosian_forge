from __future__ import absolute_import, division, print_function
import os
import sys
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.respawn import (
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils import deps

        Returns value for the specified key (removes it from user configuration).

        If an error occurs, a call will be made to AnsibleModule.fail_json.

        :param key: dconf key to reset. Should be a full path.
        :type key: str

        :returns: bool -- True if a change was made, False if no change was required.
        