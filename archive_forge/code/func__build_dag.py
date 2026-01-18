from typing import Any, Dict, Tuple, Optional
from triad.utils.convert import get_caller_global_local_vars
from fugue.dataframe import AnyDataFrame
from fugue.exceptions import FugueSQLError
from fugue.execution import AnyExecutionEngine
from fugue.execution.api import get_current_conf
from ..constants import (
from .workflow import FugueSQLWorkflow
def _build_dag(query: str, fsql_ignore_case: Optional[bool], fsql_dialect: Optional[str], args: Tuple[Any, ...], kwargs: Dict[str, Any], level: int=-2) -> FugueSQLWorkflow:
    global_vars, local_vars = get_caller_global_local_vars(start=level, end=level)
    if fsql_ignore_case is None:
        fsql_ignore_case = get_current_conf().get(FUGUE_CONF_SQL_IGNORE_CASE, False)
    if fsql_dialect is None:
        fsql_dialect = get_current_conf().get(FUGUE_CONF_SQL_DIALECT, FUGUE_SQL_DEFAULT_DIALECT)
    dag = FugueSQLWorkflow(compile_conf={FUGUE_CONF_SQL_IGNORE_CASE: fsql_ignore_case, FUGUE_CONF_SQL_DIALECT: fsql_dialect})
    try:
        dag._sql(query, global_vars, local_vars, *args, **kwargs)
    except SyntaxError as ex:
        raise SyntaxError(str(ex)).with_traceback(None) from None
    return dag