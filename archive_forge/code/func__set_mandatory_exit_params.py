from __future__ import absolute_import, division, print_function
import re
import shlex
from ansible.module_utils.basic import AnsibleModule
from collections import defaultdict, namedtuple
def _set_mandatory_exit_params(self):
    msg = '\n'.join(self._msgs)
    stdouts = '\n'.join(self._stdouts)
    stderrs = '\n'.join(self._stderrs)
    if stdouts:
        self.exit_params['stdout'] = stdouts
    if stderrs:
        self.exit_params['stderr'] = stderrs
    self.exit_params['msg'] = msg