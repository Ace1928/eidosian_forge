from plotly import exceptions, optional_imports
import plotly.colors as clrs
from plotly.figure_factory import utils
from plotly.subplots import make_subplots
import math
from numbers import Number
def _return_label(original_label, facet_labels, facet_var):
    if isinstance(facet_labels, dict):
        label = facet_labels[original_label]
    elif isinstance(facet_labels, str):
        label = '{}: {}'.format(facet_var, original_label)
    else:
        label = original_label
    return label