from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.basic import AnsibleModule
def _get_pear_path(module):
    if module.params['executable'] and os.path.isfile(module.params['executable']):
        result = module.params['executable']
    else:
        result = module.get_bin_path('pear', True, [module.params['executable']])
    return result