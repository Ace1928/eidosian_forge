from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.locale import get_best_parsable_locale
from ansible.module_utils.common.process import get_bin_path
from ansible.module_utils.common.respawn import has_respawned, probe_interpreters_for_module, respawn_module
from ansible.module_utils.facts.packages import LibMgr, CLIMgr, get_all_pkg_managers
class RPM(LibMgr):
    LIB = 'rpm'

    def list_installed(self):
        return self._lib.TransactionSet().dbMatch()

    def get_package_details(self, package):
        return dict(name=package[self._lib.RPMTAG_NAME], version=package[self._lib.RPMTAG_VERSION], release=package[self._lib.RPMTAG_RELEASE], epoch=package[self._lib.RPMTAG_EPOCH], arch=package[self._lib.RPMTAG_ARCH])

    def is_available(self):
        """ we expect the python bindings installed, but this gives warning if they are missing and we have rpm cli"""
        we_have_lib = super(RPM, self).is_available()
        try:
            get_bin_path('rpm')
            if not we_have_lib and (not has_respawned()):
                interpreters = ['/usr/libexec/platform-python', '/usr/bin/python3', '/usr/bin/python2']
                interpreter_path = probe_interpreters_for_module(interpreters, self.LIB)
                if interpreter_path:
                    respawn_module(interpreter_path)
            if not we_have_lib:
                module.warn('Found "rpm" but %s' % missing_required_lib(self.LIB))
        except ValueError:
            pass
        return we_have_lib