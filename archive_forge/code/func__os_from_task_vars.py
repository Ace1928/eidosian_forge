from __future__ import absolute_import, division, print_function
import json
from importlib import import_module
from ansible.errors import AnsibleActionFail
from ansible.module_utils._text import to_native, to_text
from ansible.module_utils.connection import Connection
from ansible.module_utils.connection import ConnectionError as AnsibleConnectionError
from ansible.plugins.action import ActionBase
from ansible_collections.ansible.utils.plugins.module_utils.common.argspec_validate import (
from ansible_collections.ansible.utils.plugins.modules.cli_parse import DOCUMENTATION
def _os_from_task_vars(self):
    """Extract an os str from the task's vars

        :return: A short OS name
        :rtype: str
        """
    os_vars = ['ansible_distribution', 'ansible_network_os']
    oper_sys = ''
    for hvar in os_vars:
        if self._task_vars.get(hvar):
            if hvar == 'ansible_network_os':
                oper_sys = self._task_vars.get(hvar, '').split('.')[-1]
                self._debug('OS set to {os}, derived from ansible_network_os'.format(os=oper_sys.lower()))
            else:
                oper_sys = self._task_vars.get(hvar)
                self._debug('OS set to {os}, using {key}'.format(os=oper_sys.lower(), key=hvar))
    return oper_sys.lower()