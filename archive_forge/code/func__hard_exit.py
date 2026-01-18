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
def _hard_exit(self, e):
    """
        There is no safe exception to return to higher level code that does not
        risk an innocent try/except finding itself executing in the wrong
        process. All code executing above WorkerProcess.run() on the stack
        conceptually belongs to another program.
        """
    try:
        display.debug(u'WORKER HARD EXIT: %s' % to_text(e))
    except BaseException:
        pass
    os._exit(1)