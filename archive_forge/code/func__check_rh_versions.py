from __future__ import (absolute_import, division, print_function)
import os
import subprocess
import ansible.module_utils.compat.typing as t
from ansible.module_utils.facts.collector import BaseFactCollector
def _check_rh_versions(self, pkg_mgr_name, collected_facts):
    if os.path.exists('/run/ostree-booted'):
        return 'atomic_container'
    pkg_mgr_name = self._default_unknown_pkg_mgr
    for bin_path in ('/usr/bin/dnf', '/usr/bin/microdnf'):
        if os.path.exists(bin_path):
            pkg_mgr_name = 'dnf5' if os.path.realpath(bin_path) == '/usr/bin/dnf5' else 'dnf'
            break
    try:
        major_version = collected_facts['ansible_distribution_major_version']
        if collected_facts['ansible_distribution'] == 'Kylin Linux Advanced Server':
            major_version = major_version.lstrip('V')
        distro_major_ver = int(major_version)
    except ValueError:
        return self._default_unknown_pkg_mgr
    if (collected_facts['ansible_distribution'] == 'Fedora' and distro_major_ver < 23 or (collected_facts['ansible_distribution'] == 'Kylin Linux Advanced Server' and distro_major_ver < 10) or (collected_facts['ansible_distribution'] == 'Amazon' and distro_major_ver < 2022) or (collected_facts['ansible_distribution'] == 'TencentOS' and distro_major_ver < 3) or (distro_major_ver < 8)) and any((pm for pm in PKG_MGRS if pm['name'] == 'yum' and os.path.exists(pm['path']))):
        pkg_mgr_name = 'yum'
    return pkg_mgr_name