from __future__ import absolute_import, division, print_function
import re
import shlex
from ansible.module_utils.basic import AnsibleModule
from collections import defaultdict, namedtuple
def _list_database(self):
    """runs pacman --sync --list with some caching"""
    if self._cached_database is None:
        dummy, packages, dummy = self.m.run_command([self.pacman_path, '--sync', '--list'], check_rc=True)
        self._cached_database = packages.splitlines()
    return self._cached_database