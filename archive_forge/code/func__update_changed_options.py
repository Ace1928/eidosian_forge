from __future__ import absolute_import, division, print_function
import time
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import bigiq_version
from ..module_utils.ipaddress import is_valid_ip
from ..module_utils.teem import send_teem
def _update_changed_options(self):
    diff = Difference(self.want, self.have)
    updatables = Parameters.updatables
    changed = dict()
    for k in updatables:
        change = diff.compare(k)
        if change is None:
            continue
        elif isinstance(change, dict):
            changed.update(change)
        else:
            changed[k] = change
    if changed:
        self.changes = UsableChanges(params=changed)
        return True
    return False