from typing import Any, Dict, List, Optional
from triad.utils.assertion import assert_or_throw
from ..collections.yielded import Yielded
from ..constants import FUGUE_CONF_WORKFLOW_EXCEPTION_INJECT
from ..dataframe import DataFrame, AnyDataFrame
from ..dataframe.api import get_native_as_df
from ..exceptions import FugueInterfacelessError, FugueWorkflowCompileError
from ..execution import make_execution_engine
from .workflow import FugueWorkflow
def _no_op_processor(df: DataFrame) -> DataFrame:
    return df