from __future__ import (absolute_import, division, print_function)
import os.path
from ansible import constants as C
from ansible.module_utils.six import string_types
from ansible.module_utils.six.moves import shlex_quote
from ansible.module_utils._text import to_text
from ansible.module_utils.common._collections_compat import MutableSequence
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.plugins.action import ActionBase
from ansible.plugins.loader import connection_loader
def _override_module_replaced_vars(self, task_vars):
    """ Some vars are substituted into the modules.  Have to make sure
        that those are correct for localhost when synchronize creates its own
        connection to localhost."""
    if 'ansible_syslog_facility' in task_vars:
        del task_vars['ansible_syslog_facility']
    for key in list(task_vars.keys()):
        if key.startswith('ansible_') and key.endswith('_interpreter'):
            del task_vars[key]
    for host in C.LOCALHOST:
        if host in task_vars['hostvars']:
            localhost = task_vars['hostvars'][host]
            break
    if 'ansible_syslog_facility' in localhost:
        task_vars['ansible_syslog_facility'] = localhost['ansible_syslog_facility']
    for key in localhost:
        if key.startswith('ansible_') and key.endswith('_interpreter'):
            task_vars[key] = localhost[key]