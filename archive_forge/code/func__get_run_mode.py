from __future__ import absolute_import, division, print_function
import copy
import glob
import os
from importlib import import_module
from ansible.errors import AnsibleActionFail, AnsibleError
from ansible.module_utils._text import to_text
from ansible.utils.display import Display
from ansible_collections.ansible.netcommon.plugins.action.network import (
def _get_run_mode(self):
    error_msg = None
    if self._config or self._running_config:
        if not self._name:
            error_msg = "'name' is required if 'config' option is set"
        if not self._state:
            error_msg = "'state' is required if 'config' option is set"
        run_mode = RunMode.RM_CONFIG
    elif self._state:
        if not self._name:
            error_msg = "'name' is required if 'state' option is set"
        run_mode = RunMode.RM_GET
    elif self._name:
        if not any([self._config, self._running_config, self._state]):
            error_msg = "If 'name' is set atleast one of 'config', 'running_config' or 'state' is required"
    else:
        run_mode = RunMode.RM_LIST
    if error_msg:
        raise AnsibleActionFail(error_msg)
    return run_mode