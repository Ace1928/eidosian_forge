import asyncio
import base64
import time
from collections import defaultdict
import numpy as np
from bokeh.models import (
from panel.io.state import set_curdoc, state
from ...core.options import CallbackError
from ...core.util import datetime_types, dimension_sanitizer, dt64_to_dt, isequal
from ...element import Table
from ...streams import (
from ...util.warnings import warn
from .util import bokeh33, convert_timestamp
class MouseLeaveCallback(PointerXYCallback):
    """
    Returns the mouse x/y-position on mouseleave event, i.e. when
    mouse leaves the plot canvas.
    """
    on_events = ['mouseleave']