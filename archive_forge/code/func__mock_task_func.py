import json
from typing import Callable, cast
from adagio.exceptions import SkippedError
from adagio.instances import (_ConfigVar, _Dependency, _DependencyDict, _Input,
from adagio.specs import ConfigSpec, InputSpec, OutputSpec, TaskSpec
from pytest import raises
from triad.collections.dict import IndexedOrderedDict, ParamDict
from triad.exceptions import InvalidOperationError
from triad.utils.hash import to_uuid
def _mock_task_func():
    pass