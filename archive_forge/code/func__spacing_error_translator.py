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
def _spacing_error_translator(e, direction, facet_arg):
    """
        Translates the spacing errors thrown by the underlying make_subplots
        routine into one that describes an argument adjustable through px.
        """
    if '%s spacing' % (direction,) in e.args[0]:
        e.args = (e.args[0] + '\nUse the {facet_arg} argument to adjust this spacing.'.format(facet_arg=facet_arg),)
        raise e