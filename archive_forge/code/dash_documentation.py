import base64
import copy
import pickle
import uuid
from collections import namedtuple
from dash.exceptions import PreventUpdate
import holoviews as hv
from holoviews.core.decollate import (
from holoviews.plotting.plotly import DynamicMap, PlotlyRenderer
from holoviews.plotting.plotly.callbacks import (
from holoviews.plotting.plotly.util import clean_internal_figure_properties
from holoviews.streams import Derived, History
import plotly.graph_objects as go
from dash import callback_context
from dash.dependencies import Input, Output, State

    Update the store with values of streams for a single type

    Args:
        store_data: Current store dictionary
        stream_event_data:  Potential stream data for current plotly event and
            traces in figures
        uid_to_streams_for_type: Mapping from trace UIDs to HoloViews streams of
            a particular type
    Returns:
        any_change: Whether any stream value has been updated
    