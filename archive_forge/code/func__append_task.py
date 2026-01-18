import json
from typing import Any, Dict, List, Optional, Type, TypeVar
from adagio.exceptions import DependencyDefinitionError, DependencyNotDefinedError
from triad.collections.dict import IndexedOrderedDict, ParamDict
from triad.exceptions import InvalidOperationError
from triad.utils.assertion import assert_or_throw as aot, assert_arg_not_none
from triad.utils.convert import (
from triad.utils.hash import to_uuid
from triad.utils.string import assert_triad_var_name
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