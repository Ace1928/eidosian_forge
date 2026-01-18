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
def _set_always_delegated_facts(self, result, task):
    """Sets host facts for ``delegate_to`` hosts for facts that should
        always be delegated

        This operation mutates ``result`` to remove the always delegated facts

        See ``ALWAYS_DELEGATE_FACT_PREFIXES``
        """
    if task.delegate_to is None:
        return
    facts = result['ansible_facts']
    always_keys = set()
    _add = always_keys.add
    for fact_key in facts:
        for always_key in ALWAYS_DELEGATE_FACT_PREFIXES:
            if fact_key.startswith(always_key):
                _add(fact_key)
    if always_keys:
        _pop = facts.pop
        always_facts = {'ansible_facts': dict(((k, _pop(k)) for k in list(facts) if k in always_keys))}
        host_list = self.get_delegated_hosts(result, task)
        _set_host_facts = self._variable_manager.set_host_facts
        for target_host in host_list:
            _set_host_facts(target_host, always_facts)