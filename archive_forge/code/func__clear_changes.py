from __future__ import absolute_import, division, print_function
import os
import time
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
def _clear_changes(self):
    redundant = ['policy_type', 'retain_inheritance_settings', 'parent_policy', 'base64', 'encoding']
    changed = {}
    for key in Parameters.returnables:
        if getattr(self.want, key) is not None and key not in redundant:
            changed[key] = getattr(self.want, key)
    if changed:
        self.changes = UsableChanges(params=changed)