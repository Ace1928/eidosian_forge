from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.locale import get_best_parsable_locale
from ansible.module_utils.common.process import get_bin_path
from ansible.module_utils.common.respawn import has_respawned, probe_interpreters_for_module, respawn_module
from ansible.module_utils.facts.packages import LibMgr, CLIMgr, get_all_pkg_managers
class PKG(CLIMgr):
    CLI = 'pkg'
    atoms = ['name', 'version', 'origin', 'installed', 'automatic', 'arch', 'category', 'prefix', 'vital']

    def list_installed(self):
        rc, out, err = module.run_command([self._cli, 'query', '%%%s' % '\t%'.join(['n', 'v', 'R', 't', 'a', 'q', 'o', 'p', 'V'])])
        if rc != 0 or err:
            raise Exception('Unable to list packages rc=%s : %s' % (rc, err))
        return out.splitlines()

    def get_package_details(self, package):
        pkg = dict(zip(self.atoms, package.split('\t')))
        if 'arch' in pkg:
            try:
                pkg['arch'] = pkg['arch'].split(':')[2]
            except IndexError:
                pass
        if 'automatic' in pkg:
            pkg['automatic'] = bool(int(pkg['automatic']))
        if 'category' in pkg:
            pkg['category'] = pkg['category'].split('/', 1)[0]
        if 'version' in pkg:
            if ',' in pkg['version']:
                pkg['version'], pkg['port_epoch'] = pkg['version'].split(',', 1)
            else:
                pkg['port_epoch'] = 0
            if '_' in pkg['version']:
                pkg['version'], pkg['revision'] = pkg['version'].split('_', 1)
            else:
                pkg['revision'] = '0'
        if 'vital' in pkg:
            pkg['vital'] = bool(int(pkg['vital']))
        return pkg