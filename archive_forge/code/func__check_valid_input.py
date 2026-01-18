from typing import Any, Dict, List, Optional
from triad.utils.assertion import assert_or_throw
from ..collections.yielded import Yielded
from ..constants import FUGUE_CONF_WORKFLOW_EXCEPTION_INJECT
from ..dataframe import DataFrame, AnyDataFrame
from ..dataframe.api import get_native_as_df
from ..exceptions import FugueInterfacelessError, FugueWorkflowCompileError
from ..execution import make_execution_engine
from .workflow import FugueWorkflow
def _check_valid_input(df: Any, save_path: Optional[str]) -> None:
    if isinstance(df, str):
        assert_or_throw('.csv' not in df and '.json' not in df, FugueInterfacelessError('Fugue transform can only load parquet file paths.\n                Csv and json are disallowed'))
    if save_path:
        assert_or_throw('.csv' not in save_path and '.json' not in save_path, FugueInterfacelessError('Fugue transform can only load parquet file paths.\n                Csv and json are disallowed'))