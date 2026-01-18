from __future__ import annotations
import json
import urllib.parse
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Union, cast
from typing_extensions import TypeAlias
from streamlit import type_util
from streamlit.elements.lib.streamlit_plotly_theme import (
from streamlit.errors import StreamlitAPIException
from streamlit.proto.PlotlyChart_pb2 import PlotlyChart as PlotlyChartProto
from streamlit.runtime.legacy_caching import caching
from streamlit.runtime.metrics_util import gather_metrics
@caching.cache
def _plot_to_url_or_load_cached_url(*args: Any, **kwargs: Any) -> go.Figure:
    """Call plotly.plot wrapped in st.cache.

    This is so we don't unnecessarily upload data to Plotly's SASS if nothing
    changed since the previous upload.
    """
    try:
        import chart_studio.plotly as ply
    except ImportError:
        import plotly.plotly as ply
    return ply.plot(*args, **kwargs)