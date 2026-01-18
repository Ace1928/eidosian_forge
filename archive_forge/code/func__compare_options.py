from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.crypto.plugins.module_utils.version import LooseVersion
from ansible_collections.community.crypto.plugins.module_utils.openssh.backends.common import (
from ansible_collections.community.crypto.plugins.module_utils.openssh.certificate import (
def _compare_options(self):
    try:
        critical_options, extensions = parse_option_list(self.options)
    except ValueError as e:
        return self.module.fail_json(msg=to_native(e))
    return all([set(self.original_data.critical_options) == set(critical_options), set(self.original_data.extensions) == set(extensions)])