from __future__ import annotations
import importlib
import warnings
from packaging.version import Version
def _dask_expr_enabled() -> bool:
    import pandas as pd
    import dask
    use_dask_expr = dask.config.get('dataframe.query-planning')
    if use_dask_expr is False or (use_dask_expr is None and Version(pd.__version__).major < 2):
        return False
    try:
        import dask_expr
    except ImportError:
        msg = '\nDask dataframe query planning is disabled because dask-expr is not installed.\n\nYou can install it with `pip install dask[dataframe]` or `conda install dask`.\nThis will raise in a future version.\n'
        if use_dask_expr is None:
            warnings.warn(msg, FutureWarning)
            return False
        else:
            raise ImportError(msg)
    return True