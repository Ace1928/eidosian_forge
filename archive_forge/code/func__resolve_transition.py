from collections import OrderedDict
import copy
from functools import partial, reduce
import inspect
import logging
from six import string_types
from ..core import State, Machine, Transition, Event, listify, MachineError, EventData
def _resolve_transition(self, event_data):
    dst_name_path = self.dest.split(event_data.machine.state_cls.separator)
    _ = event_data.machine.get_state(dst_name_path)
    state_tree = event_data.machine.build_state_tree(listify(getattr(event_data.model, event_data.machine.model_attribute)), event_data.machine.state_cls.separator)
    scope = event_data.machine.get_global_name(join=False)
    tmp_tree = state_tree.get(dst_name_path[0], None)
    root = []
    while tmp_tree is not None:
        root.append(dst_name_path.pop(0))
        tmp_tree = tmp_tree.get(dst_name_path[0], None) if len(dst_name_path) > 0 else None
    if not dst_name_path:
        dst_name_path = [root.pop()]
    scoped_tree = reduce(dict.get, scope + root, state_tree)
    exit_partials = [partial(event_data.machine.get_state(root + state_name).scoped_exit, event_data, scope + root + state_name[:-1]) for state_name in resolve_order(scoped_tree)]
    if dst_name_path:
        new_states, enter_partials = self._enter_nested(root, dst_name_path, scope + root, event_data)
    else:
        new_states, enter_partials = ({}, [])
    scoped_tree.clear()
    for new_key, value in new_states.items():
        scoped_tree[new_key] = value
        break
    return (state_tree, exit_partials, enter_partials)