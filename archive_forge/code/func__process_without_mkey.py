from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import _load_params
import datetime
from ansible.module_utils.six import raise_from
def _process_without_mkey(self):
    if self.module.params['state'] == 'absent':
        self.module.fail_json(msg="This module doesn't not support state:absent yet.")
    return self.create_object()