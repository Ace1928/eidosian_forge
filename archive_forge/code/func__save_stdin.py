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
def _save_stdin(self):
    self._new_stdin = None
    try:
        if sys.stdin.isatty() and sys.stdin.fileno() is not None:
            try:
                self._new_stdin = os.fdopen(os.dup(sys.stdin.fileno()))
            except OSError:
                pass
    except (AttributeError, ValueError):
        pass
    if self._new_stdin is None:
        self._new_stdin = open(os.devnull)