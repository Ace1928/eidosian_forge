from __future__ import absolute_import, division, print_function
import os
import re
import sys
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.urls import fetch_file
from ansible.module_utils.compat.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.locale import get_best_parsable_locale
from ansible.module_utils.common.respawn import has_respawned, probe_interpreters_for_module, respawn_module
from ansible.module_utils.yumdnf import YumDnf, yumdnf_argument_spec
def _package_dict(self, package):
    """Return a dictionary of information for the package."""
    result = {'name': package.name, 'arch': package.arch, 'epoch': str(package.epoch), 'release': package.release, 'version': package.version, 'repo': package.repoid}
    result['envra'] = '{epoch}:{name}-{version}-{release}.{arch}'.format(**result)
    result['nevra'] = result['envra']
    if package.installtime == 0:
        result['yumstate'] = 'available'
    else:
        result['yumstate'] = 'installed'
    return result