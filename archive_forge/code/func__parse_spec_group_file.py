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
def _parse_spec_group_file(self):
    pkg_specs, grp_specs, module_specs, filenames = ([], [], [], [])
    already_loaded_comps = False
    for name in self.names:
        if '://' in name:
            name = fetch_file(self.module, name)
            filenames.append(name)
        elif name.endswith('.rpm'):
            filenames.append(name)
        elif name.startswith('/'):
            pkg_spec = self._whatprovides(name)
            if pkg_spec:
                pkg_specs.append(pkg_spec)
                continue
        elif name.startswith('@') or '/' in name:
            if not already_loaded_comps:
                self.base.read_comps()
                already_loaded_comps = True
            grp_env_mdl_candidate = name[1:].strip()
            if self.with_modules:
                mdl = self.module_base._get_modules(grp_env_mdl_candidate)
                if mdl[0]:
                    module_specs.append(grp_env_mdl_candidate)
                else:
                    grp_specs.append(grp_env_mdl_candidate)
            else:
                grp_specs.append(grp_env_mdl_candidate)
        else:
            pkg_specs.append(name)
    return (pkg_specs, grp_specs, module_specs, filenames)