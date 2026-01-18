from typing import Any, Dict, Tuple
import panel as _pn
from . import hvPlotTabular, post_patch
from .util import _fugue_ipython

            Process the dataframes and output the result as
            a pn.Column.

            Parameters:
            -----------
            dfs: fugue.DataFrames
            