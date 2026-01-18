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
@debug_closure
def _process_pending_results(self, iterator, one_pass=False, max_passes=None):
    """
        Reads results off the final queue and takes appropriate action
        based on the result (executing callbacks, updating state, etc.).
        """
    ret_results = []
    cur_pass = 0
    while True:
        try:
            self._results_lock.acquire()
            task_result = self._results.popleft()
        except IndexError:
            break
        finally:
            self._results_lock.release()
        original_host = task_result._host
        original_task = task_result._task
        role_ran = False
        if task_result.is_failed():
            role_ran = True
            ignore_errors = original_task.ignore_errors
            if not ignore_errors:
                state_when_failed = iterator.get_state_for_host(original_host.name)
                display.debug('marking %s as failed' % original_host.name)
                if original_task.run_once:
                    for h in self._inventory.get_hosts(iterator._play.hosts):
                        if h.name not in self._tqm._unreachable_hosts:
                            iterator.mark_host_failed(h)
                else:
                    iterator.mark_host_failed(original_host)
                state, dummy = iterator.get_next_task_for_host(original_host, peek=True)
                if iterator.is_failed(original_host) and state and (state.run_state == IteratingStates.COMPLETE):
                    self._tqm._failed_hosts[original_host.name] = True
                if iterator.is_any_block_rescuing(state_when_failed):
                    self._tqm._stats.increment('rescued', original_host.name)
                    iterator._play._removed_hosts.remove(original_host.name)
                    self._variable_manager.set_nonpersistent_facts(original_host.name, dict(ansible_failed_task=wrap_var(original_task.serialize()), ansible_failed_result=task_result._result))
                else:
                    self._tqm._stats.increment('failures', original_host.name)
            else:
                self._tqm._stats.increment('ok', original_host.name)
                self._tqm._stats.increment('ignored', original_host.name)
                if 'changed' in task_result._result and task_result._result['changed']:
                    self._tqm._stats.increment('changed', original_host.name)
            self._tqm.send_callback('v2_runner_on_failed', task_result, ignore_errors=ignore_errors)
        elif task_result.is_unreachable():
            ignore_unreachable = original_task.ignore_unreachable
            if not ignore_unreachable:
                self._tqm._unreachable_hosts[original_host.name] = True
                iterator._play._removed_hosts.append(original_host.name)
                self._tqm._stats.increment('dark', original_host.name)
            else:
                self._tqm._stats.increment('ok', original_host.name)
                self._tqm._stats.increment('ignored', original_host.name)
            self._tqm.send_callback('v2_runner_on_unreachable', task_result)
        elif task_result.is_skipped():
            self._tqm._stats.increment('skipped', original_host.name)
            self._tqm.send_callback('v2_runner_on_skipped', task_result)
        else:
            role_ran = True
            if original_task.loop:
                result_items = task_result._result.get('results', [])
            else:
                result_items = [task_result._result]
            for result_item in result_items:
                if '_ansible_notify' in result_item and task_result.is_changed():
                    host_state = iterator.get_state_for_host(original_host.name)
                    for notification in result_item['_ansible_notify']:
                        handler = Sentinel
                        for handler in self.search_handlers_by_notification(notification, iterator):
                            if host_state.run_state == IteratingStates.HANDLERS:
                                if handler.notify_host(original_host):
                                    self._tqm.send_callback('v2_playbook_on_notify', handler, original_host)
                            else:
                                iterator.add_notification(original_host.name, notification)
                                display.vv(f'Notification for handler {notification} has been saved.')
                                break
                        if handler is Sentinel:
                            msg = f"The requested handler '{notification}' was not found in either the main handlers list nor in the listening handlers list"
                            if C.ERROR_ON_MISSING_HANDLER:
                                raise AnsibleError(msg)
                            else:
                                display.warning(msg)
                if 'add_host' in result_item:
                    new_host_info = result_item.get('add_host', dict())
                    self._inventory.add_dynamic_host(new_host_info, result_item)
                    if result_item.get('changed') and new_host_info['host_name'] not in self._hosts_cache_all:
                        self._hosts_cache_all.append(new_host_info['host_name'])
                elif 'add_group' in result_item:
                    self._inventory.add_dynamic_group(original_host, result_item)
                if 'add_host' in result_item or 'add_group' in result_item:
                    item_vars = _get_item_vars(result_item, original_task)
                    found_task_vars = self._queued_task_cache.get((original_host.name, task_result._task._uuid))['task_vars']
                    if item_vars:
                        all_task_vars = combine_vars(found_task_vars, item_vars)
                    else:
                        all_task_vars = found_task_vars
                    all_task_vars[original_task.register] = wrap_var(result_item)
                    post_process_whens(result_item, original_task, Templar(self._loader), all_task_vars)
                    if original_task.loop or original_task.loop_with:
                        new_item_result = TaskResult(task_result._host, task_result._task, result_item, task_result._task_fields)
                        self._tqm.send_callback('v2_runner_item_on_ok', new_item_result)
                        if result_item.get('changed', False):
                            task_result._result['changed'] = True
                        if result_item.get('failed', False):
                            task_result._result['failed'] = True
                if 'ansible_facts' in result_item and original_task.action not in C._ACTION_DEBUG:
                    if original_task.delegate_to is not None and original_task.delegate_facts:
                        host_list = self.get_delegated_hosts(result_item, original_task)
                    else:
                        self._set_always_delegated_facts(result_item, original_task)
                        host_list = self.get_task_hosts(iterator, original_host, original_task)
                    if original_task.action in C._ACTION_INCLUDE_VARS:
                        for var_name, var_value in result_item['ansible_facts'].items():
                            for target_host in host_list:
                                self._variable_manager.set_host_variable(target_host, var_name, var_value)
                    else:
                        cacheable = result_item.pop('_ansible_facts_cacheable', False)
                        for target_host in host_list:
                            is_set_fact = original_task.action in C._ACTION_SET_FACT
                            if not is_set_fact or cacheable:
                                self._variable_manager.set_host_facts(target_host, result_item['ansible_facts'].copy())
                            if is_set_fact:
                                self._variable_manager.set_nonpersistent_facts(target_host, result_item['ansible_facts'].copy())
                if 'ansible_stats' in result_item and 'data' in result_item['ansible_stats'] and result_item['ansible_stats']['data']:
                    if 'per_host' not in result_item['ansible_stats'] or result_item['ansible_stats']['per_host']:
                        host_list = self.get_task_hosts(iterator, original_host, original_task)
                    else:
                        host_list = [None]
                    data = result_item['ansible_stats']['data']
                    aggregate = 'aggregate' in result_item['ansible_stats'] and result_item['ansible_stats']['aggregate']
                    for myhost in host_list:
                        for k in data.keys():
                            if aggregate:
                                self._tqm._stats.update_custom_stats(k, data[k], myhost)
                            else:
                                self._tqm._stats.set_custom_stats(k, data[k], myhost)
            if 'diff' in task_result._result:
                if self._diff or getattr(original_task, 'diff', False):
                    self._tqm.send_callback('v2_on_file_diff', task_result)
            if not isinstance(original_task, TaskInclude):
                self._tqm._stats.increment('ok', original_host.name)
                if 'changed' in task_result._result and task_result._result['changed']:
                    self._tqm._stats.increment('changed', original_host.name)
            self._tqm.send_callback('v2_runner_on_ok', task_result)
        if original_task.register:
            if not isidentifier(original_task.register):
                raise AnsibleError("Invalid variable name in 'register' specified: '%s'" % original_task.register)
            host_list = self.get_task_hosts(iterator, original_host, original_task)
            clean_copy = strip_internal_keys(module_response_deepcopy(task_result._result))
            if 'invocation' in clean_copy:
                del clean_copy['invocation']
            for target_host in host_list:
                self._variable_manager.set_nonpersistent_facts(target_host, {original_task.register: clean_copy})
        self._pending_results -= 1
        if original_host.name in self._blocked_hosts:
            del self._blocked_hosts[original_host.name]
        if original_task._role is not None and role_ran:
            role_obj = self._get_cached_role(original_task, iterator._play)
            role_obj._had_task_run[original_host.name] = True
        ret_results.append(task_result)
        if one_pass or (max_passes is not None and cur_pass + 1 >= max_passes):
            break
        cur_pass += 1
    return ret_results