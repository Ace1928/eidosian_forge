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
def _human_format(number):
    units = ['', 'K', 'M', 'G', 'T', 'P']
    k = 1000.0
    magnitude = int(floor(log(number, k)))
    return '%.2f%s' % (number / k ** magnitude, units[magnitude])