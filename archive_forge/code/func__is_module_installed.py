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
def _is_module_installed(self, module_spec):
    if self.with_modules:
        module_spec = module_spec.strip()
        module_list, nsv = self.module_base._get_modules(module_spec)
        enabled_streams = self.base._moduleContainer.getEnabledStream(nsv.name)
        if enabled_streams:
            if nsv.stream:
                if nsv.stream in enabled_streams:
                    return True
                else:
                    return False
            else:
                return True
    return False