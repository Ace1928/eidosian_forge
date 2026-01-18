import json
from typing import Any, Dict, List, Optional, Type, TypeVar
from adagio.exceptions import DependencyDefinitionError, DependencyNotDefinedError
from triad.collections.dict import IndexedOrderedDict, ParamDict
from triad.exceptions import InvalidOperationError
from triad.utils.assertion import assert_or_throw as aot, assert_arg_not_none
from triad.utils.convert import (
from triad.utils.hash import to_uuid
from triad.utils.string import assert_triad_var_name
def _validate_dependency(self):
    if set(self.node_spec.dependency.keys()) != set(self.inputs.keys()):
        raise DependencyNotDefinedError(self.name + ' input', self.inputs.keys(), self.node_spec.dependency.keys())
    for k, v in self.node_spec.dependency.items():
        t = v.split('.', 1)
        if len(t) == 1:
            aot(t[0] in self.parent_workflow.inputs, lambda: f'{t[0]} is not an input of the workflow')
            self.inputs[k].validate_spec(self.parent_workflow.inputs[t[0]])
        else:
            aot(t[0] != self.name, lambda: f'{v} tries to connect to self node {self.name}')
            task = self.parent_workflow.tasks[t[0]]
            aot(t[1] in task.outputs, lambda: f'{t[1]} is not an output of node {task.name}')
            self.inputs[k].validate_spec(task.outputs[t[1]])