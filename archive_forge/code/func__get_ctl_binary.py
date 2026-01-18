from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
def _get_ctl_binary(module):
    for command in ['apache2ctl', 'apachectl']:
        ctl_binary = module.get_bin_path(command)
        if ctl_binary is not None:
            return ctl_binary
    module.fail_json(msg='Neither of apache2ctl nor apachectl found. At least one apache control binary is necessary.')