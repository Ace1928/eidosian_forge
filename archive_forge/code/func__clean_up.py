from __future__ import (absolute_import, division, print_function)
import os
import sys
import traceback
from jinja2.exceptions import TemplateNotFound
from multiprocessing.queues import Queue
from ansible.errors import AnsibleConnectionFailure, AnsibleError
from ansible.executor.task_executor import TaskExecutor
from ansible.module_utils.common.text.converters import to_text
from ansible.utils.display import Display
from ansible.utils.multiprocessing import context as multiprocessing_context
def _clean_up(self):
    self._loader.cleanup_all_tmp_files()