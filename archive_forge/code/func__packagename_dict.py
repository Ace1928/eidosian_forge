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
def _packagename_dict(self, packagename):
    """
        Return a dictionary of information for a package name string or None
        if the package name doesn't contain at least all NVR elements
        """
    if packagename[-4:] == '.rpm':
        packagename = packagename[:-4]
    rpm_nevr_re = re.compile('(\\S+)-(?:(\\d*):)?(.*)-(~?\\w+[\\w.+]*)')
    try:
        arch = None
        nevr, arch = self._split_package_arch(packagename)
        if arch:
            packagename = nevr
        rpm_nevr_match = rpm_nevr_re.match(packagename)
        if rpm_nevr_match:
            name, epoch, version, release = rpm_nevr_re.match(packagename).groups()
            if not version or not version.split('.')[0].isdigit():
                return None
        else:
            return None
    except AttributeError as e:
        self.module.fail_json(msg='Error attempting to parse package: %s, %s' % (packagename, to_native(e)), rc=1, results=[])
    if not epoch:
        epoch = '0'
    if ':' in name:
        epoch_name = name.split(':')
        epoch = epoch_name[0]
        name = ''.join(epoch_name[1:])
    result = {'name': name, 'epoch': epoch, 'release': release, 'version': version}
    return result