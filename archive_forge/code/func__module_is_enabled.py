from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
def _module_is_enabled(module):
    control_binary = _get_ctl_binary(module)
    result, stdout, stderr = module.run_command([control_binary, '-M'])
    if result != 0:
        error_msg = 'Error executing %s: %s' % (control_binary, stderr)
        if module.params['ignore_configcheck']:
            if 'AH00534' in stderr and 'mpm_' in module.params['name']:
                if module.params['warn_mpm_absent']:
                    module.warnings.append('No MPM module loaded! apache2 reload AND other module actions will fail if no MPM module is loaded immediately.')
            else:
                module.warnings.append(error_msg)
            return False
        else:
            module.fail_json(msg=error_msg)
    searchstring = ' ' + module.params['identifier']
    return searchstring in stdout