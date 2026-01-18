from __future__ import (absolute_import, division, print_function)
import os
import time
import typing as t
from ansible import constants as C
from ansible.executor.module_common import get_action_args_with_defaults
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.plugins.action import ActionBase
from ansible.utils.vars import merge_hash
def _get_module_args(self, fact_module: str, task_vars: dict[str, t.Any]) -> dict[str, t.Any]:
    mod_args = self._task.args.copy()
    if fact_module not in C._ACTION_SETUP:
        try:
            name = self._connection.ansible_name.removeprefix('ansible.netcommon.')
        except AttributeError:
            name = self._connection._load_name.split('.')[-1]
        if name not in ('network_cli', 'httpapi', 'netconf'):
            subset = mod_args.pop('gather_subset', None)
            if subset not in ('all', ['all'], None):
                self._display.warning('Not passing subset(%s) to %s' % (subset, fact_module))
        timeout = mod_args.pop('gather_timeout', None)
        if timeout is not None:
            self._display.warning('Not passing timeout(%s) to %s' % (timeout, fact_module))
        fact_filter = mod_args.pop('filter', None)
        if fact_filter is not None:
            self._display.warning('Not passing filter(%s) to %s' % (fact_filter, fact_module))
    mod_args = dict(((k, v) for k, v in mod_args.items() if v is not None))
    resolved_fact_module = self._shared_loader_obj.module_loader.find_plugin_with_context(fact_module, collection_list=self._task.collections).resolved_fqcn
    mod_args = get_action_args_with_defaults(resolved_fact_module, mod_args, self._task.module_defaults, self._templar, action_groups=self._task._parent._play._action_groups)
    return mod_args