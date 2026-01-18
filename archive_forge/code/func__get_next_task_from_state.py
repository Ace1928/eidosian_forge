from __future__ import (absolute_import, division, print_function)
import fnmatch
from enum import IntEnum, IntFlag
from ansible import constants as C
from ansible.errors import AnsibleAssertionError
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.playbook.block import Block
from ansible.playbook.task import Task
from ansible.utils.display import Display
def _get_next_task_from_state(self, state, host):
    task = None
    while True:
        try:
            block = state._blocks[state.cur_block]
        except IndexError:
            state.run_state = IteratingStates.COMPLETE
            return (state, None)
        if state.run_state == IteratingStates.SETUP:
            if not state.pending_setup:
                state.pending_setup = True
                gathering = C.DEFAULT_GATHERING
                implied = self._play.gather_facts is None or boolean(self._play.gather_facts, strict=False)
                if gathering == 'implicit' and implied or (gathering == 'explicit' and boolean(self._play.gather_facts, strict=False)) or (gathering == 'smart' and implied and (not self._variable_manager._fact_cache.get(host.name, {}).get('_ansible_facts_gathered', False))):
                    setup_block = self._blocks[0]
                    if setup_block.has_tasks() and len(setup_block.block) > 0:
                        task = setup_block.block[0]
            else:
                state.pending_setup = False
                state.run_state = IteratingStates.TASKS
                if not state.did_start_at_task:
                    state.cur_block += 1
                    state.cur_regular_task = 0
                    state.cur_rescue_task = 0
                    state.cur_always_task = 0
                    state.tasks_child_state = None
                    state.rescue_child_state = None
                    state.always_child_state = None
        elif state.run_state == IteratingStates.TASKS:
            if state.pending_setup:
                state.pending_setup = False
            if state.tasks_child_state:
                state.tasks_child_state, task = self._get_next_task_from_state(state.tasks_child_state, host=host)
                if self._check_failed_state(state.tasks_child_state):
                    state.tasks_child_state = None
                    self._set_failed_state(state)
                elif task is None or state.tasks_child_state.run_state == IteratingStates.COMPLETE:
                    state.tasks_child_state = None
                    continue
            elif self._check_failed_state(state):
                state.run_state = IteratingStates.RESCUE
            elif state.cur_regular_task >= len(block.block):
                state.run_state = IteratingStates.ALWAYS
            else:
                task = block.block[state.cur_regular_task]
                if isinstance(task, Block):
                    state.tasks_child_state = HostState(blocks=[task])
                    state.tasks_child_state.run_state = IteratingStates.TASKS
                    task = None
                state.cur_regular_task += 1
        elif state.run_state == IteratingStates.RESCUE:
            if state.rescue_child_state:
                state.rescue_child_state, task = self._get_next_task_from_state(state.rescue_child_state, host=host)
                if self._check_failed_state(state.rescue_child_state):
                    state.rescue_child_state = None
                    self._set_failed_state(state)
                elif task is None or state.rescue_child_state.run_state == IteratingStates.COMPLETE:
                    state.rescue_child_state = None
                    continue
            elif state.fail_state & FailedStates.RESCUE == FailedStates.RESCUE:
                state.run_state = IteratingStates.ALWAYS
            elif state.cur_rescue_task >= len(block.rescue):
                if len(block.rescue) > 0:
                    state.fail_state = FailedStates.NONE
                state.run_state = IteratingStates.ALWAYS
                state.did_rescue = True
            else:
                task = block.rescue[state.cur_rescue_task]
                if isinstance(task, Block):
                    state.rescue_child_state = HostState(blocks=[task])
                    state.rescue_child_state.run_state = IteratingStates.TASKS
                    task = None
                state.cur_rescue_task += 1
        elif state.run_state == IteratingStates.ALWAYS:
            if state.always_child_state:
                state.always_child_state, task = self._get_next_task_from_state(state.always_child_state, host=host)
                if self._check_failed_state(state.always_child_state):
                    state.always_child_state = None
                    self._set_failed_state(state)
                elif task is None or state.always_child_state.run_state == IteratingStates.COMPLETE:
                    state.always_child_state = None
                    continue
            elif state.cur_always_task >= len(block.always):
                if state.fail_state != FailedStates.NONE:
                    state.run_state = IteratingStates.COMPLETE
                else:
                    state.cur_block += 1
                    state.cur_regular_task = 0
                    state.cur_rescue_task = 0
                    state.cur_always_task = 0
                    state.run_state = IteratingStates.TASKS
                    state.tasks_child_state = None
                    state.rescue_child_state = None
                    state.always_child_state = None
                    state.did_rescue = False
            else:
                task = block.always[state.cur_always_task]
                if isinstance(task, Block):
                    state.always_child_state = HostState(blocks=[task])
                    state.always_child_state.run_state = IteratingStates.TASKS
                    task = None
                state.cur_always_task += 1
        elif state.run_state == IteratingStates.HANDLERS:
            if state.update_handlers:
                state.handlers = self.handlers[:]
                state.update_handlers = False
                state.cur_handlers_task = 0
            while True:
                try:
                    task = state.handlers[state.cur_handlers_task]
                except IndexError:
                    task = None
                    state.run_state = state.pre_flushing_run_state
                    state.update_handlers = True
                    break
                else:
                    state.cur_handlers_task += 1
                    if task.is_host_notified(host):
                        break
        elif state.run_state == IteratingStates.COMPLETE:
            return (state, None)
        if task:
            break
    return (state, task)