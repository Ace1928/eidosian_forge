from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.locale import get_best_parsable_locale
from ansible.module_utils.common.process import get_bin_path
from ansible.module_utils.common.respawn import has_respawned, probe_interpreters_for_module, respawn_module
from ansible.module_utils.facts.packages import LibMgr, CLIMgr, get_all_pkg_managers
class PORTAGE(CLIMgr):
    CLI = 'qlist'
    atoms = ['category', 'name', 'version', 'ebuild_revision', 'slots', 'prefixes', 'sufixes']

    def list_installed(self):
        rc, out, err = module.run_command(' '.join([self._cli, '-Iv', '|', 'xargs', '-n', '1024', 'qatom']), use_unsafe_shell=True)
        if rc != 0:
            raise RuntimeError('Unable to list packages rc=%s : %s' % (rc, to_native(err)))
        return out.splitlines()

    def get_package_details(self, package):
        return dict(zip(self.atoms, package.split()))