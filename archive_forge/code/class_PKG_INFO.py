from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.locale import get_best_parsable_locale
from ansible.module_utils.common.process import get_bin_path
from ansible.module_utils.common.respawn import has_respawned, probe_interpreters_for_module, respawn_module
from ansible.module_utils.facts.packages import LibMgr, CLIMgr, get_all_pkg_managers
class PKG_INFO(CLIMgr):
    CLI = 'pkg_info'

    def list_installed(self):
        rc, out, err = module.run_command([self._cli, '-a'])
        if rc != 0 or err:
            raise Exception('Unable to list packages rc=%s : %s' % (rc, err))
        return out.splitlines()

    def get_package_details(self, package):
        raw_pkg_details = {'name': package, 'version': ''}
        details = package.split(maxsplit=1)[0].rsplit('-', maxsplit=1)
        try:
            return {'name': details[0], 'version': details[1]}
        except IndexError:
            return raw_pkg_details