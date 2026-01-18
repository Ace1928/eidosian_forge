from __future__ import (absolute_import, division, print_function)
from ansible import constants as C
from ansible.errors import AnsibleError, AnsibleAssertionError, AnsibleParserError
from ansible.executor.play_iterator import IteratingStates, FailedStates
from ansible.module_utils.common.text.converters import to_text
from ansible.playbook.handler import Handler
from ansible.playbook.included_file import IncludedFile
from ansible.playbook.task import Task
from ansible.plugins.loader import action_loader
from ansible.plugins.strategy import StrategyBase
from ansible.template import Templar
from ansible.utils.display import Display
def _get_next_task_lockstep(self, hosts, iterator):
    """
        Returns a list of (host, task) tuples, where the task may
        be a noop task to keep the iterator in lock step across
        all hosts.
        """
    noop_task = Task()
    noop_task.action = 'meta'
    noop_task.args['_raw_params'] = 'noop'
    noop_task.implicit = True
    noop_task.set_loader(iterator._play._loader)
    state_task_per_host = {}
    for host in hosts:
        state, task = iterator.get_next_task_for_host(host, peek=True)
        if task is not None:
            state_task_per_host[host] = (state, task)
    if not state_task_per_host:
        return [(h, None) for h in hosts]
    if self._in_handlers and (not any(filter(lambda rs: rs == IteratingStates.HANDLERS, (s.run_state for s, dummy in state_task_per_host.values())))):
        self._in_handlers = False
    if self._in_handlers:
        lowest_cur_handler = min((s.cur_handlers_task for s, t in state_task_per_host.values() if s.run_state == IteratingStates.HANDLERS))
    else:
        task_uuids = [t._uuid for s, t in state_task_per_host.values()]
        _loop_cnt = 0
        while _loop_cnt <= 1:
            try:
                cur_task = iterator.all_tasks[iterator.cur_task]
            except IndexError:
                iterator.cur_task = 0
                _loop_cnt += 1
            else:
                iterator.cur_task += 1
                if cur_task._uuid in task_uuids:
                    break
        else:
            raise AnsibleAssertionError('BUG: There seems to be a mismatch between tasks in PlayIterator and HostStates.')
    host_tasks = []
    for host, (state, task) in state_task_per_host.items():
        if self._in_handlers and lowest_cur_handler == state.cur_handlers_task or (not self._in_handlers and cur_task._uuid == task._uuid):
            iterator.set_state_for_host(host.name, state)
            host_tasks.append((host, task))
        else:
            host_tasks.append((host, noop_task))
    if not self._in_handlers and cur_task.action in C._ACTION_META and (cur_task.args.get('_raw_params') == 'flush_handlers'):
        self._in_handlers = True
    return host_tasks