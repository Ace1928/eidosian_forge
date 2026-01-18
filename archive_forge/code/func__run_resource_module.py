from __future__ import absolute_import, division, print_function
import copy
import glob
import os
from importlib import import_module
from ansible.errors import AnsibleActionFail, AnsibleError
from ansible.module_utils._text import to_text
from ansible.utils.display import Display
from ansible_collections.ansible.netcommon.plugins.action.network import (
def _run_resource_module(self, prefix_os_name=False):
    new_task = self._task.copy()
    self._module = self._get_resource_module(prefix_os_name=prefix_os_name)
    if not self._module:
        msg = "Could not find resource module '%s' for os name '%s'" % (self._name, self._os_name)
        raise AnsibleActionFail(msg)
    new_task.action = self._module
    action = self._shared_loader_obj.action_loader.get(self._rm_play_context.network_os, task=new_task, connection=self._connection, play_context=self._rm_play_context, loader=self._loader, templar=self._templar, shared_loader_obj=self._shared_loader_obj)
    display.vvvv('Running resource module %s' % self._module)
    for option in ['os_name', 'name']:
        if option in new_task.args:
            new_task.args.pop(option)
    result = action.run(task_vars=self._task_vars)
    result.update({'resource_module_name': self._module})
    return result