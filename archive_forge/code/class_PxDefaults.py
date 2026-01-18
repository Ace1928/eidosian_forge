import plotly.graph_objs as go
import plotly.io as pio
from collections import namedtuple, OrderedDict
from ._special_inputs import IdentityMap, Constant, Range
from .trendline_functions import ols, lowess, rolling, expanding, ewm
from _plotly_utils.basevalidators import ColorscaleValidator
from plotly.colors import qualitative, sequential
import math
from packaging import version
import pandas as pd
import numpy as np
from plotly._subplots import (
class PxDefaults(object):
    __slots__ = ['template', 'width', 'height', 'color_discrete_sequence', 'color_discrete_map', 'color_continuous_scale', 'symbol_sequence', 'symbol_map', 'line_dash_sequence', 'line_dash_map', 'pattern_shape_sequence', 'pattern_shape_map', 'size_max', 'category_orders', 'labels']

    def __init__(self):
        self.reset()

    def reset(self):
        self.template = None
        self.width = None
        self.height = None
        self.color_discrete_sequence = None
        self.color_discrete_map = {}
        self.color_continuous_scale = None
        self.symbol_sequence = None
        self.symbol_map = {}
        self.line_dash_sequence = None
        self.line_dash_map = {}
        self.pattern_shape_sequence = None
        self.pattern_shape_map = {}
        self.size_max = 20
        self.category_orders = {}
        self.labels = {}