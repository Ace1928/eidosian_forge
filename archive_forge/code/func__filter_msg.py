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
def _filter_msg(self, msg, ids):
    """
        Filter event values that do not originate from the plotting
        handles associated with a particular stream using their
        ids to match them.
        """
    filtered_msg = {}
    for k, v in msg.items():
        if isinstance(v, dict) and 'id' in v:
            if v['id'] in ids:
                filtered_msg[k] = v['value']
        else:
            filtered_msg[k] = v
    return filtered_msg