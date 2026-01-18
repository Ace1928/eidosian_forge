from __future__ import absolute_import, division, print_function
import os
import re
from ansible.module_utils.basic import AnsibleModule
def _check_installed_pkg(module, package, repository_path):
    """
    Check the package on AIX.
    It verifies if the package is installed and information

    :param module: Ansible module parameters spec.
    :param package: Package/fileset name.
    :param repository_path: Repository package path.
    :return: Bool, package data.
    """
    lslpp_cmd = module.get_bin_path('lslpp', True)
    rc, lslpp_result, err = module.run_command('%s -lcq %s*' % (lslpp_cmd, package))
    if rc == 1:
        package_state = ' '.join(err.split()[-2:])
        if package_state == 'not installed.':
            return (False, None)
        else:
            module.fail_json(msg='Failed to run lslpp.', rc=rc, err=err)
    if rc != 0:
        module.fail_json(msg='Failed to run lslpp.', rc=rc, err=err)
    pkg_data = {}
    full_pkg_data = lslpp_result.splitlines()
    for line in full_pkg_data:
        pkg_name, fileset, level = line.split(':')[0:3]
        pkg_data[pkg_name] = (fileset, level)
    return (True, pkg_data)