from __future__ import (absolute_import, division, print_function)
import os
import subprocess
import ansible.module_utils.compat.typing as t
from ansible.module_utils.facts.collector import BaseFactCollector
class OpenBSDPkgMgrFactCollector(BaseFactCollector):
    name = 'pkg_mgr'
    _fact_ids = set()
    _platform = 'OpenBSD'

    def collect(self, module=None, collected_facts=None):
        return {'pkg_mgr': 'openbsd_pkg'}