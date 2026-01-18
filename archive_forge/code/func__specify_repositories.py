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
def _specify_repositories(self, base, disablerepo, enablerepo):
    """Enable and disable repositories matching the provided patterns."""
    base.read_all_repos()
    repos = base.repos
    for repo_pattern in disablerepo:
        if repo_pattern:
            for repo in repos.get_matching(repo_pattern):
                repo.disable()
    for repo_pattern in enablerepo:
        if repo_pattern:
            for repo in repos.get_matching(repo_pattern):
                repo.enable()