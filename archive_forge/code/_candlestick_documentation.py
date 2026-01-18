from plotly.figure_factory import utils
from plotly.figure_factory._ohlc import (
from plotly.graph_objs import graph_objs

        Separate increasing data from decreasing data.

        The data is increasing when close value > open value
        and decreasing when the close value <= open value.
        