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
def _install_remote_rpms(self, filenames):
    if int(dnf.__version__.split('.')[0]) >= 2:
        pkgs = list(sorted(self.base.add_remote_rpms(list(filenames)), reverse=True))
    else:
        pkgs = []
        try:
            for filename in filenames:
                pkgs.append(self.base.add_remote_rpm(filename))
        except IOError as e:
            if to_text('Can not load RPM file') in to_text(e):
                self.module.fail_json(msg='Error occurred attempting remote rpm install of package: {0}. {1}'.format(filename, to_native(e)), results=[], rc=1)
    if self.update_only:
        self._update_only(pkgs)
    else:
        for pkg in pkgs:
            try:
                if self._is_newer_version_installed(self._package_dict(pkg)['nevra']):
                    if self.allow_downgrade:
                        self.base.package_install(pkg, strict=self.base.conf.strict)
                else:
                    self.base.package_install(pkg, strict=self.base.conf.strict)
            except Exception as e:
                self.module.fail_json(msg='Error occurred attempting remote rpm operation: {0}'.format(to_native(e)), results=[], rc=1)