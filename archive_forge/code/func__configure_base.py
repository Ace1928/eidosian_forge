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
def _configure_base(self, base, conf_file, disable_gpg_check, installroot='/', sslverify=True):
    """Configure the dnf Base object."""
    conf = base.conf
    if conf_file:
        if not os.access(conf_file, os.R_OK):
            self.module.fail_json(msg='cannot read configuration file', conf_file=conf_file, results=[])
        else:
            conf.config_file_path = conf_file
    conf.read()
    conf.debuglevel = 0
    conf.gpgcheck = not disable_gpg_check
    conf.localpkg_gpgcheck = not disable_gpg_check
    conf.assumeyes = True
    conf.sslverify = sslverify
    conf.installroot = installroot
    conf.substitutions.update_from_etc(installroot)
    if self.exclude:
        _excludes = list(conf.exclude)
        _excludes.extend(self.exclude)
        conf.exclude = _excludes
    if self.disable_excludes:
        _disable_excludes = list(conf.disable_excludes)
        if self.disable_excludes not in _disable_excludes:
            _disable_excludes.append(self.disable_excludes)
            conf.disable_excludes = _disable_excludes
    if self.releasever is not None:
        conf.substitutions['releasever'] = self.releasever
    if conf.substitutions.get('releasever') is None:
        self.module.warn('Unable to detect release version (use "releasever" option to specify release version)')
        conf.substitutions['releasever'] = ''
    if self.skip_broken:
        conf.strict = 0
    if self.nobest:
        conf.best = 0
    if self.download_only:
        conf.downloadonly = True
        if self.download_dir:
            conf.destdir = self.download_dir
    if self.cacheonly:
        conf.cacheonly = True
    conf.clean_requirements_on_remove = self.autoremove
    conf.install_weak_deps = self.install_weak_deps