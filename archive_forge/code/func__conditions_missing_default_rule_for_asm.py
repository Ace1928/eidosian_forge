from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.six import iteritems
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import compare_complex_list
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
def _conditions_missing_default_rule_for_asm(self, want_actions):
    if want_actions is None:
        actions = self.have.actions
    else:
        actions = want_actions
    if actions is None:
        return False
    if any((x for x in actions if x['type'] == 'enable')):
        conditions = self._diff_complex_items(self.want.conditions, self.have.conditions)
        if conditions is None:
            return False
        if any((y for y in conditions if y['type'] not in ['all_traffic', 'http_uri', 'http_host', 'tcp'])):
            return True
    return False