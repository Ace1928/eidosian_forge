import json
from typing import Callable, cast
from adagio.exceptions import SkippedError
from adagio.instances import (_ConfigVar, _Dependency, _DependencyDict, _Input,
from adagio.specs import ConfigSpec, InputSpec, OutputSpec, TaskSpec
from pytest import raises
from triad.collections.dict import IndexedOrderedDict, ParamDict
from triad.exceptions import InvalidOperationError
from triad.utils.hash import to_uuid
class MockTaskForVar(_Task):

    def __init__(self):
        self.spec = MockSpec()

    @property
    def name(self) -> str:
        return 'taskname'

    def __uuid__(self) -> str:
        return 'id'