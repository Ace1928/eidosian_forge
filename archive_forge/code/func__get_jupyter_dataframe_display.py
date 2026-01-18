import html
import json
from typing import Any, Dict, List, Optional
from IPython import get_ipython
from IPython.core.magic import Magics, cell_magic, magics_class, needs_local_scope
from IPython.display import HTML, display
from triad import ParamDict
from triad.utils.convert import to_instance
from triad.utils.pyarrow import _field_to_expression
from fugue import DataFrame, DataFrameDisplay, ExecutionEngine
from fugue import fsql as fugue_sql
from fugue import get_dataset_display, make_execution_engine
from fugue.dataframe import YieldedDataFrame
from fugue.exceptions import FugueSQLSyntaxError
@get_dataset_display.candidate(lambda ds: get_ipython() is not None and isinstance(ds, DataFrame), priority=3.0)
def _get_jupyter_dataframe_display(ds: DataFrame):
    return JupyterDataFrameDisplay(ds)