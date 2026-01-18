from __future__ import (absolute_import, division, print_function)
import os
import sys
import tempfile
import threading
import time
import typing as t
import multiprocessing.queues
from ansible import constants as C
from ansible import context
from ansible.errors import AnsibleError
from ansible.executor.play_iterator import PlayIterator
from ansible.executor.stats import AggregateStats
from ansible.executor.task_result import TaskResult
from ansible.module_utils.six import string_types
from ansible.module_utils.common.text.converters import to_text, to_native
from ansible.playbook.play_context import PlayContext
from ansible.playbook.task import Task
from ansible.plugins.loader import callback_loader, strategy_loader, module_loader
from ansible.plugins.callback import CallbackBase
from ansible.template import Templar
from ansible.vars.hostvars import HostVars
from ansible.vars.reserved import warn_if_reserved
from ansible.utils.display import Display
from ansible.utils.lock import lock_decorator
from ansible.utils.multiprocessing import context as multiprocessing_context
from dataclasses import dataclass
def _cleanup_processes(self):
    if hasattr(self, '_workers'):
        for attempts_remaining in range(C.WORKER_SHUTDOWN_POLL_COUNT - 1, -1, -1):
            if not any((worker_prc and worker_prc.is_alive() for worker_prc in self._workers)):
                break
            if attempts_remaining:
                time.sleep(C.WORKER_SHUTDOWN_POLL_DELAY)
            else:
                display.warning('One or more worker processes are still running and will be terminated.')
        for worker_prc in self._workers:
            if worker_prc and worker_prc.is_alive():
                try:
                    worker_prc.terminate()
                except AttributeError:
                    pass