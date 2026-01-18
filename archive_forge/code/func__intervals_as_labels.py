import io
import numpy as np
import os
import pandas as pd
import warnings
from math import log, floor
from numbers import Number
from plotly import optional_imports
import plotly.colors as clrs
from plotly.figure_factory import utils
from plotly.exceptions import PlotlyError
import plotly.graph_objs as go
def _intervals_as_labels(array_of_intervals, round_legend_values, exponent_format):
    """
    Transform an number interval to a clean string for legend

    Example: [-inf, 30] to '< 30'
    """
    infs = [float('-inf'), float('inf')]
    string_intervals = []
    for interval in array_of_intervals:
        if round_legend_values:
            rnd_interval = [int(interval[i]) if interval[i] not in infs else interval[i] for i in range(2)]
        else:
            rnd_interval = [round(interval[0], 2), round(interval[1], 2)]
        num0 = rnd_interval[0]
        num1 = rnd_interval[1]
        if exponent_format:
            if num0 not in infs:
                num0 = _human_format(num0)
            if num1 not in infs:
                num1 = _human_format(num1)
        else:
            if num0 not in infs:
                num0 = '{:,}'.format(num0)
            if num1 not in infs:
                num1 = '{:,}'.format(num1)
        if num0 == float('-inf'):
            as_str = '< {}'.format(num1)
        elif num1 == float('inf'):
            as_str = '> {}'.format(num0)
        else:
            as_str = '{} - {}'.format(num0, num1)
        string_intervals.append(as_str)
    return string_intervals