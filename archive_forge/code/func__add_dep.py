from typing import Any, List, Optional, Dict, Union, Iterable
from adagio.instances import TaskContext, WorkflowContext
from adagio.specs import InputSpec, OutputSpec, TaskSpec, WorkflowSpec
from triad.utils.assertion import assert_or_throw
from triad.utils.hash import to_uuid
from triad.collections import ParamDict
from qpd.qpd_engine import QPDEngine
from qpd.dataframe import Column, DataFrame, DataFrames
def _add_dep(self, deps: Dict[str, str], obj: Any):
    if isinstance(obj, QPDTask):
        oe = obj.name + '._0'
    elif isinstance(obj, QPDTaskWrapper):
        oe = obj.task.name + '._0'
    else:
        raise ValueError(f'{obj} is invalid')
    deps[f'_{len(deps)}'] = oe