from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.locale import get_best_parsable_locale
from ansible.module_utils.common.process import get_bin_path
from ansible.module_utils.common.respawn import has_respawned, probe_interpreters_for_module, respawn_module
from ansible.module_utils.facts.packages import LibMgr, CLIMgr, get_all_pkg_managers
class APT(LibMgr):
    LIB = 'apt'

    def __init__(self):
        self._cache = None
        super(APT, self).__init__()

    @property
    def pkg_cache(self):
        if self._cache is not None:
            return self._cache
        self._cache = self._lib.Cache()
        return self._cache

    def is_available(self):
        """ we expect the python bindings installed, but if there is apt/apt-get give warning about missing bindings"""
        we_have_lib = super(APT, self).is_available()
        if not we_have_lib:
            for exe in ('apt', 'apt-get', 'aptitude'):
                try:
                    get_bin_path(exe)
                except ValueError:
                    continue
                else:
                    if not has_respawned():
                        interpreters = ['/usr/bin/python3', '/usr/bin/python2']
                        interpreter_path = probe_interpreters_for_module(interpreters, self.LIB)
                        if interpreter_path:
                            respawn_module(interpreter_path)
                    module.warn('Found "%s" but %s' % (exe, missing_required_lib('apt')))
                    break
        return we_have_lib

    def list_installed(self):
        cache = self.pkg_cache
        return [pk for pk in cache.keys() if cache[pk].is_installed]

    def get_package_details(self, package):
        ac_pkg = self.pkg_cache[package].installed
        return dict(name=package, version=ac_pkg.version, arch=ac_pkg.architecture, category=ac_pkg.section, origin=ac_pkg.origins[0].origin)