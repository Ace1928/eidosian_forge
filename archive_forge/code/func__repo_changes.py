from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.six.moves import configparser, StringIO
from io import open
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
def _repo_changes(module, realrepo, repocmp):
    """Check whether the 2 given repos have different settings."""
    for k in repocmp:
        if repocmp[k] and k not in realrepo:
            return True
    for k, v in realrepo.items():
        if k in repocmp and repocmp[k]:
            valold = str(repocmp[k] or '')
            valnew = v or ''
            if k == 'url':
                if '$releasever' in valold or '$releasever' in valnew:
                    cmd = ['rpm', '-q', '--qf', '%{version}', '-f', '/etc/os-release']
                    rc, stdout, stderr = module.run_command(cmd, check_rc=True)
                    valnew = valnew.replace('$releasever', stdout)
                    valold = valold.replace('$releasever', stdout)
                if '$basearch' in valold or '$basearch' in valnew:
                    cmd = ['rpm', '-q', '--qf', '%{arch}', '-f', '/etc/os-release']
                    rc, stdout, stderr = module.run_command(cmd, check_rc=True)
                    valnew = valnew.replace('$basearch', stdout)
                    valold = valold.replace('$basearch', stdout)
                valold, valnew = (valold.rstrip('/'), valnew.rstrip('/'))
            if valold != valnew:
                return True
    return False