import math
from ..bindings.view_state import ViewState
from .type_checking import is_pandas_df
def _squared_diff(x, x0):
    return (x0 - x) * (x0 - x)