import copy
import inspect
from typing import Any, Callable, Dict, Iterable, Optional
from triad import extension_method
from triad.collections.function_wrapper import (
from triad.utils.assertion import assert_or_throw
from triad.utils.convert import get_caller_global_local_vars, to_function
from fugue.constants import FUGUE_ENTRYPOINT
from fugue.exceptions import FugueInterfacelessError
from fugue.workflow.workflow import FugueWorkflow, WorkflowDataFrame, WorkflowDataFrames
def _infer_workflow(self, *args: Any, **kwargs: Any) -> Optional[FugueWorkflow]:

    def select_args() -> Iterable[Any]:
        for a in args:
            if isinstance(a, (WorkflowDataFrames, WorkflowDataFrame)):
                yield a
        for _, v in kwargs.items():
            if isinstance(v, (WorkflowDataFrames, WorkflowDataFrame)):
                yield v
    wf: Optional[FugueWorkflow] = None
    for a in select_args():
        if isinstance(a, WorkflowDataFrame):
            assert_or_throw(wf is None or a.workflow is wf, ValueError('different parenet workflows found on input dataframes'))
            wf = a.workflow
        elif isinstance(a, WorkflowDataFrames):
            for k, v in a.items():
                assert_or_throw(isinstance(v, WorkflowDataFrame), lambda: ValueError(f'{k}:{v} is not a WorkflowDataFrame'))
                assert_or_throw(wf is None or v.workflow is wf, ValueError('different parenet workflows found on input dataframes'))
                wf = v.workflow
    return wf