from __future__ import (absolute_import, division, print_function)
import datetime
import json
import copy
from functools import partial
from ansible.inventory.host import Host
from ansible.module_utils._text import to_text
from ansible.parsing.ajson import AnsibleJSONEncoder
from ansible.plugins.callback import CallbackBase
def _record_task_result(self, event_name, on_info, result, **kwargs):
    """This function is used as a partial to add failed/skipped info in a single method"""
    host = result._host
    task = result._task
    result_copy = result._result.copy()
    result_copy.update(on_info)
    result_copy['action'] = task.action
    task_result = self._find_result_task(host, task)
    end_time = current_time()
    task_result['task']['duration']['end'] = end_time
    self.results[-1]['play']['duration']['end'] = end_time
    task_result_copy = copy.deepcopy(task_result)
    task_result_copy['hosts'][host.name] = result_copy
    if not self._is_lockstep:
        key = (host.get_name(), task._uuid)
        del self._task_map[key]
    self._write_event(event_name, task_result_copy)