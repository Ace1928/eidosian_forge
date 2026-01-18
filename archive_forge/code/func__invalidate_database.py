from __future__ import absolute_import, division, print_function
import re
import shlex
from ansible.module_utils.basic import AnsibleModule
from collections import defaultdict, namedtuple
def _invalidate_database(self):
    """invalidates the pacman --sync --list cache"""
    self._cached_database = None