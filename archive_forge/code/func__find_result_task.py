from __future__ import (absolute_import, division, print_function)
import datetime
import json
import copy
from functools import partial
from ansible.inventory.host import Host
from ansible.module_utils._text import to_text
from ansible.parsing.ajson import AnsibleJSONEncoder
from ansible.plugins.callback import CallbackBase
def _find_result_task(self, host, task):
    key = (host.get_name(), task._uuid)
    return self._task_map.get(key, self.results[-1]['tasks'][-1])