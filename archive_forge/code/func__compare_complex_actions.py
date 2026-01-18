from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.six import iteritems
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import compare_complex_list
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
def _compare_complex_actions(self):
    types = ['insert', 'remove', 'replace']
    if self.want.actions:
        want = [item for item in self.want.actions if item['type'] in types]
        have = [item for item in self.have.actions if item['type'] in types]
        result = compare_complex_list(want, have)
        if result:
            return True
    return False