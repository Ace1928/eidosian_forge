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
class GlyphDrawCallback(CDSCallback):
    _style_callback = '\n      var types = Bokeh.require("core/util/types");\n      var changed = false\n      for (var i = 0; i < cb_obj.length; i++) {\n        for (var style in styles) {\n          var value = styles[style];\n          if (types.isArray(value)) {\n            value = value[i % value.length];\n          }\n          if (cb_obj.data[style][i] !== value) {\n            cb_obj.data[style][i] = value;\n            changed = true;\n          }\n        }\n      }\n      if (changed)\n        cb_obj.change.emit()\n    '

    def _create_style_callback(self, cds, glyph):
        stream = self.streams[0]
        col = cds.column_names[0]
        length = len(cds.data[col])
        for style, values in stream.styles.items():
            cds.data[style] = [values[i % len(values)] for i in range(length)]
            setattr(glyph, style, style)
        cb = CustomJS(code=self._style_callback, args={'styles': stream.styles, 'empty': stream.empty_value})
        cds.js_on_change('data', cb)

    def _update_cds_vdims(self, data):
        """
        Add any value dimensions not already in the data ensuring the
        element can be reconstituted in entirety.
        """
        element = self.plot.current_frame
        stream = self.streams[0]
        for d in element.vdims:
            dim = dimension_sanitizer(d.name)
            if dim in data:
                continue
            values = element.dimension_values(d)
            if len(values) != len(next(iter(data.values()))):
                values = np.concatenate([values, [stream.empty_value]])
            data[dim] = values