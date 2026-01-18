from __future__ import absolute_import, division, print_function
import os
import re
import tempfile
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.common.text.converters import to_bytes
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems, string_types
def _upgrade_casks(self):
    for cask in self.casks:
        self.current_cask = cask
        self._upgrade_current_cask()
    return True