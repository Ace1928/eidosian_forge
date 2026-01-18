from __future__ import annotations
import datetime as dt
from typing import (
import numpy as np
import param
from bokeh.models import CustomJS
from bokeh.models.formatters import TickFormatter
from bokeh.models.widgets import (
from bokeh.models.widgets.sliders import NumericalSlider as _BkNumericalSlider
from param.parameterized import resolve_value
from ..config import config
from ..io import state
from ..io.resources import CDN_DIST
from ..layout import Column, Panel, Row
from ..util import (
from ..viewable import Layoutable
from ..widgets import FloatInput, IntInput
from .base import CompositeWidget, Widget
from .input import StaticText
class ContinuousSlider(_SliderBase):
    format = param.ClassSelector(class_=(str, TickFormatter), doc='\n        A custom format string or Bokeh TickFormatter.')
    _supports_embed: ClassVar[bool] = True
    __abstract = True

    def __init__(self, **params):
        if 'value' not in params:
            params['value'] = params.get('start', self.start)
        super().__init__(**params)

    def _get_embed_state(self, root, values=None, max_opts=3):
        ref = root.ref['id']
        w_model, parent = self._models[ref]
        if not isinstance(w_model, _BkNumericalSlider):
            is_composite = True
            parent = w_model
            w_model = w_model.select_one({'type': _BkNumericalSlider})
        else:
            is_composite = False
        _, _, doc, comm = state._views[ref]
        start, end, step = (w_model.start, w_model.end, w_model.step)
        if values is None:
            span = end - start
            dtype = int if isinstance(step, int) else float
            if span / step > max_opts - 1:
                step = dtype(span / (max_opts - 1))
            values = [dtype(v) for v in np.arange(start, end + step, step)]
        elif any((v < start or v > end for v in values)):
            raise ValueError('Supplied embed states for %s widget outside of valid range.' % type(self).__name__)
        layout_opts = {k: v for k, v in self.param.values().items() if k in Layoutable.param and k != 'name'}
        if is_composite:
            layout_opts['show_value'] = False
        else:
            layout_opts['name'] = self.name
        value = values[np.argmin(np.abs(np.array(values) - self.value))]
        dw = DiscreteSlider(options=values, value=value, **layout_opts)
        dw.link(self, value='value')
        self._models.pop(ref)
        index = parent.children.index(w_model)
        with config.set(embed=True):
            w_model = dw._get_model(doc, root, parent, comm)
        link = CustomJS(code=dw._jslink.code['value'], args={'source': w_model.children[1], 'target': w_model.children[0]})
        parent.children[index] = w_model
        w_model = w_model.children[1]
        w_model.js_on_change('value', link)
        return (dw, w_model, values, lambda x: x.value, 'value', 'cb_obj.value')