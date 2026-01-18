import json
from typing import Any, Dict, List, Optional, Type, TypeVar
from adagio.exceptions import DependencyDefinitionError, DependencyNotDefinedError
from triad.collections.dict import IndexedOrderedDict, ParamDict
from triad.exceptions import InvalidOperationError
from triad.utils.assertion import assert_or_throw as aot, assert_arg_not_none
from triad.utils.convert import (
from triad.utils.hash import to_uuid
from triad.utils.string import assert_triad_var_name
class _NodeSpec(object):

    def __init__(self, workflow: 'WorkflowSpec', name: str, dependency: Optional[Dict[str, str]], config: Optional[Dict[str, Any]], config_dependency: Optional[Dict[str, str]]):
        self.workflow = workflow
        self.name = name
        self.dependency = dependency or {}
        self.config = config or {}
        self.config_dependency = config_dependency or {}

    def __uuid__(self) -> str:
        return to_uuid(self.dependency, self.config, self.config_dependency)

    @property
    def jsondict(self) -> ParamDict:
        return dict(name=self.name, dependency=self.dependency, config=self.config, config_dependency=self.config_dependency)