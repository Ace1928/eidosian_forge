from __future__ import (absolute_import, division, print_function)
import fnmatch
from enum import IntEnum, IntFlag
from ansible import constants as C
from ansible.errors import AnsibleAssertionError
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.playbook.block import Block
from ansible.playbook.task import Task
from ansible.utils.display import Display
def _check_failed_state(self, state):
    if state is None:
        return False
    elif state.run_state == IteratingStates.RESCUE and self._check_failed_state(state.rescue_child_state):
        return True
    elif state.run_state == IteratingStates.ALWAYS and self._check_failed_state(state.always_child_state):
        return True
    elif state.fail_state != FailedStates.NONE:
        if state.run_state == IteratingStates.RESCUE and state.fail_state & FailedStates.RESCUE == 0:
            return False
        elif state.run_state == IteratingStates.ALWAYS and state.fail_state & FailedStates.ALWAYS == 0:
            return False
        else:
            return not (state.did_rescue and state.fail_state & FailedStates.ALWAYS == 0)
    elif state.run_state == IteratingStates.TASKS and self._check_failed_state(state.tasks_child_state):
        cur_block = state._blocks[state.cur_block]
        if len(cur_block.rescue) > 0 and state.fail_state & FailedStates.RESCUE == 0:
            return False
        else:
            return True
    return False