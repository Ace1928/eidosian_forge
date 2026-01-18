from __future__ import absolute_import, division, print_function
import hashlib
import os
import re
import uuid
from ansible.errors import AnsibleError
from ansible.module_utils._text import to_bytes, to_text
from ansible.module_utils.connection import Connection, ConnectionError
from ansible.module_utils.six.moves.urllib.parse import urlsplit
from ansible.plugins.action import ActionBase
from ansible.utils.display import Display
def _get_network_os(self, task_vars):
    if 'network_os' in self._task.args and self._task.args['network_os']:
        display.vvvv('Getting network OS from task argument')
        network_os = self._task.args['network_os']
    elif self._play_context.network_os:
        display.vvvv('Getting network OS from inventory')
        network_os = self._play_context.network_os
    elif 'network_os' in task_vars.get('ansible_facts', {}) and task_vars['ansible_facts']['network_os']:
        display.vvvv('Getting network OS from fact')
        network_os = task_vars['ansible_facts']['network_os']
    else:
        raise AnsibleError('ansible_network_os must be specified on this host')
    return network_os