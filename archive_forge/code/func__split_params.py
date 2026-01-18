from typing import Any, Dict, Tuple
from triad.utils.assertion import assert_or_throw
from triad.utils.convert import get_caller_global_local_vars
from fugue._utils.misc import import_fsql_dependency
from ..collections.yielded import Yielded
from ..constants import FUGUE_CONF_SQL_DIALECT, FUGUE_CONF_SQL_IGNORE_CASE
from ..dataframe.api import is_df
from ..dataframe.dataframe import DataFrame
from ..workflow.workflow import FugueWorkflow, WorkflowDataFrame, WorkflowDataFrames
from ._utils import LazyWorkflowDataFrame, fill_sql_template
def _split_params(self, params: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, LazyWorkflowDataFrame]]:
    p: Dict[str, Any] = {}
    dfs: Dict[str, LazyWorkflowDataFrame] = {}
    for k, v in params.items():
        if isinstance(v, (int, str, float, bool)):
            p[k] = v
        elif isinstance(v, (DataFrame, Yielded)) or is_df(v):
            dfs[k] = LazyWorkflowDataFrame(k, v, self)
        else:
            p[k] = v
    return (p, dfs)