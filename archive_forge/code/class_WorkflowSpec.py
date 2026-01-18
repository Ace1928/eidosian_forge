import json
from typing import Any, Dict, List, Optional, Type, TypeVar
from adagio.exceptions import DependencyDefinitionError, DependencyNotDefinedError
from triad.collections.dict import IndexedOrderedDict, ParamDict
from triad.exceptions import InvalidOperationError
from triad.utils.assertion import assert_or_throw as aot, assert_arg_not_none
from triad.utils.convert import (
from triad.utils.hash import to_uuid
from triad.utils.string import assert_triad_var_name
class WorkflowSpec(TaskSpec):

    def __init__(self, configs: Any=None, inputs: Any=None, outputs: Any=None, metadata=None, deterministic: bool=True, lazy: bool=True, tasks: Optional[List[Any]]=None, internal_dependency: Optional[Dict[str, str]]=None):
        super().__init__(configs, inputs, outputs, _no_op, metadata=metadata, deterministic=deterministic, lazy=lazy)
        self.tasks: IndexedOrderedDict[str, TaskSpec] = {}
        self.internal_dependency: Dict[str, str] = {}
        if tasks is not None:
            for t in tasks:
                self._append_task(to_taskspec(t, self))
        if internal_dependency is not None:
            for k, v in internal_dependency.items():
                self.link(k, v)

    def __uuid__(self) -> str:
        return to_uuid(super().__uuid__(), self.tasks, self.internal_dependency)

    def add_task(self, name: str, task: Any, dependency: Optional[Dict[str, str]]=None, config: Optional[Dict[str, Any]]=None, config_dependency: Optional[Dict[str, str]]=None) -> TaskSpec:
        _t = to_taskspec(task)
        aot(_t._node_spec is None, 'node_spec must not be set')
        _t._node_spec = _NodeSpec(self, name, dependency, config, config_dependency)
        return self._append_task(_t)

    def link(self, output: str, to_expr: str):
        try:
            aot(output in self.outputs, lambda: f'{output} is not an output of the workflow')
            aot(output not in self.internal_dependency, lambda: f'{output} is already defined')
            t = to_expr.split('.', 1)
            if len(t) == 1:
                aot(t[0] in self.inputs, lambda: f'{t[0]} is not an input of the workflow')
                self.outputs[output].validate_spec(self.inputs[t[0]])
            else:
                node = self.tasks[t[0]]
                aot(t[1] in node.outputs, lambda: f'{t[1]} is not an output of {node}')
                self.outputs[output].validate_spec(node.outputs[t[1]])
            self.internal_dependency[output] = to_expr
        except Exception as e:
            raise DependencyDefinitionError(e)

    @property
    def jsondict(self) -> ParamDict:
        d = super().jsondict
        d['tasks'] = [x.jsondict for x in self.tasks.values()]
        d['internal_dependency'] = self.internal_dependency
        del d['func']
        return d

    def validate(self) -> None:
        if set(self.outputs.keys()) != set(self.internal_dependency.keys()):
            raise DependencyNotDefinedError('workflow output', self.outputs.keys(), self.internal_dependency.keys())

    def _append_task(self, task: TaskSpec) -> TaskSpec:
        name = task.name
        assert_triad_var_name(name)
        aot(name not in self.tasks, lambda: KeyError(f'{name} already exists in workflow'))
        aot(task.parent_workflow is self, lambda: InvalidOperationError(f'{task} has mismatching node_spec'))
        try:
            task._validate_config()
            task._validate_dependency()
        except DependencyDefinitionError:
            raise
        except Exception as e:
            raise DependencyDefinitionError(e)
        self.tasks[name] = task
        return task