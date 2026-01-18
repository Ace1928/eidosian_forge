from __future__ import (absolute_import, division, print_function)
import os
import subprocess
import ansible.module_utils.compat.typing as t
from ansible.module_utils.facts.collector import BaseFactCollector
def _check_apt_flavor(self, pkg_mgr_name):
    rpm_query = '/usr/bin/rpm -q --whatprovides /usr/bin/apt-get'.split()
    if os.path.exists('/usr/bin/rpm'):
        with open(os.devnull, 'w') as null:
            try:
                subprocess.check_call(rpm_query, stdout=null, stderr=null)
                pkg_mgr_name = 'apt_rpm'
            except subprocess.CalledProcessError:
                pkg_mgr_name = 'apt'
    return pkg_mgr_name