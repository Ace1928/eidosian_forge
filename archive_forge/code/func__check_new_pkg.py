from __future__ import absolute_import, division, print_function
import os
import re
from ansible.module_utils.basic import AnsibleModule
def _check_new_pkg(module, package, repository_path):
    """
    Check if the package of fileset is correct name and repository path.

    :param module: Ansible module arguments spec.
    :param package: Package/fileset name.
    :param repository_path: Repository package path.
    :return: Bool, package information.
    """
    if os.path.isdir(repository_path):
        installp_cmd = module.get_bin_path('installp', True)
        rc, package_result, err = module.run_command('%s -l -MR -d %s' % (installp_cmd, repository_path))
        if rc != 0:
            module.fail_json(msg='Failed to run installp.', rc=rc, err=err)
        if package == 'all':
            pkg_info = 'All packages on dir'
            return (True, pkg_info)
        else:
            pkg_info = {}
            for line in package_result.splitlines():
                if re.findall(package, line):
                    pkg_name = line.split()[0].strip()
                    pkg_version = line.split()[1].strip()
                    pkg_info[pkg_name] = pkg_version
                    return (True, pkg_info)
        return (False, None)
    else:
        module.fail_json(msg='Repository path %s is not valid.' % repository_path)