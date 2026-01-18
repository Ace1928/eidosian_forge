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
class CDSCallback(Callback):
    """
    A Stream callback that syncs the data on a bokeh ColumnDataSource
    model with Python.
    """
    attributes = {'data': 'source.data'}
    models = ['source']
    on_changes = ['data', 'patching']

    def initialize(self, plot_id=None):
        super().initialize(plot_id)
        plot = self.plot
        data = self._process_msg({'data': plot.handles['source'].data})['data']
        for stream in self.streams:
            stream.update(data=data)

    def _process_msg(self, msg):
        if 'data' not in msg:
            return {}
        msg['data'] = dict(msg['data'])
        for col, values in msg['data'].items():
            if isinstance(values, dict):
                shape = values.pop('shape', None)
                dtype = values.pop('dtype', None)
                values.pop('dimension', None)
                items = sorted([(int(k), v) for k, v in values.items()])
                values = [v for k, v in items]
                if dtype is not None:
                    values = np.array(values, dtype=dtype).reshape(shape)
            elif isinstance(values, list) and values and isinstance(values[0], dict):
                new_values = []
                for vals in values:
                    if isinstance(vals, dict):
                        shape = vals.pop('shape', None)
                        dtype = vals.pop('dtype', None)
                        vals.pop('dimension', None)
                        vals = sorted([(int(k), v) for k, v in vals.items()])
                        vals = [v for k, v in vals]
                        if dtype is not None:
                            vals = np.array(vals, dtype=dtype).reshape(shape)
                    new_values.append(vals)
                values = new_values
            elif any((isinstance(v, (int, float)) for v in values)):
                values = [np.nan if v is None else v for v in values]
            elif isinstance(values, list) and len(values) == 4 and (values[2] in ('big', 'little')) and isinstance(values[3], list):
                buffer = base64.decodebytes(values[0].encode())
                dtype = np.dtype(values[1]).newbyteorder(values[2])
                values = np.frombuffer(buffer, dtype)
            msg['data'][col] = values
        return self._transform(msg)