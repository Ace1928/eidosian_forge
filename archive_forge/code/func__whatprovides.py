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
def _whatprovides(self, filepath):
    self.base.read_all_repos()
    available = self.base.sack.query().available()
    files_filter = available.filter(file=filepath)
    pkg_spec = files_filter.union(available.filter(provides=filepath)).run()
    if pkg_spec:
        return pkg_spec[0].name