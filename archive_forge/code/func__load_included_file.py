from __future__ import (absolute_import, division, print_function)
import cmd
import functools
import os
import pprint
import queue
import sys
import threading
import time
import typing as t
from collections import deque
from multiprocessing import Lock
from jinja2.exceptions import UndefinedError
from ansible import constants as C
from ansible import context
from ansible.errors import AnsibleError, AnsibleFileNotFound, AnsibleUndefinedVariable, AnsibleParserError
from ansible.executor import action_write_locks
from ansible.executor.play_iterator import IteratingStates, PlayIterator
from ansible.executor.process.worker import WorkerProcess
from ansible.executor.task_result import TaskResult
from ansible.executor.task_queue_manager import CallbackSend, DisplaySend, PromptSend
from ansible.module_utils.six import string_types
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.connection import Connection, ConnectionError
from ansible.playbook.conditional import Conditional
from ansible.playbook.handler import Handler
from ansible.playbook.helpers import load_list_of_blocks
from ansible.playbook.task import Task
from ansible.playbook.task_include import TaskInclude
from ansible.plugins import loader as plugin_loader
from ansible.template import Templar
from ansible.utils.display import Display
from ansible.utils.fqcn import add_internal_fqcns
from ansible.utils.unsafe_proxy import wrap_var
from ansible.utils.sentinel import Sentinel
from ansible.utils.vars import combine_vars, isidentifier
from ansible.vars.clean import strip_internal_keys, module_response_deepcopy
def _load_included_file(self, included_file, iterator, is_handler=False):
    """
        Loads an included YAML file of tasks, applying the optional set of variables.

        Raises AnsibleError exception in case of a failure during including a file,
        in such case the caller is responsible for marking the host(s) as failed
        using PlayIterator.mark_host_failed().
        """
    display.debug('loading included file: %s' % included_file._filename)
    try:
        data = self._loader.load_from_file(included_file._filename)
        if data is None:
            return []
        elif not isinstance(data, list):
            raise AnsibleError('included task files must contain a list of tasks')
        ti_copy = self._copy_included_file(included_file)
        block_list = load_list_of_blocks(data, play=iterator._play, parent_block=ti_copy.build_parent_block(), role=included_file._task._role, use_handlers=is_handler, loader=self._loader, variable_manager=self._variable_manager)
        for host in included_file._hosts:
            self._tqm._stats.increment('ok', host.name)
    except AnsibleParserError:
        raise
    except AnsibleError as e:
        if isinstance(e, AnsibleFileNotFound):
            reason = "Could not find or access '%s' on the Ansible Controller." % to_text(e.file_name)
        else:
            reason = to_text(e)
        for r in included_file._results:
            r._result['failed'] = True
        for host in included_file._hosts:
            tr = TaskResult(host=host, task=included_file._task, return_data=dict(failed=True, reason=reason))
            self._tqm._stats.increment('failures', host.name)
            self._tqm.send_callback('v2_runner_on_failed', tr)
        raise AnsibleError(reason) from e
    self._tqm.send_callback('v2_playbook_on_include', included_file)
    display.debug('done processing included file')
    return block_list