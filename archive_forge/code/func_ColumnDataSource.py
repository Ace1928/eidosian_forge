import re
import numpy as np
from pandas import DataFrame
from ...rcparams import rcParams
def ColumnDataSource(*args, **kwargs):
    """Wrap bokeh.models.ColumnDataSource."""
    from bokeh.models import ColumnDataSource
    return ColumnDataSource(*args, **kwargs)