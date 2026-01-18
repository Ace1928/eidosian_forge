import base64
from io import BytesIO
import panel as pn
import param
from param.parameterized import bothmethod
from ...core import HoloMap
from ...core.options import Store
from ..renderer import HTML_TAGS, MIME_TYPES, Renderer
from .callbacks import callbacks
from .util import clean_internal_figure_properties
def _PlotlyHoloviewsPane(fig_dict, **kwargs):
    """
    Custom Plotly pane constructor for use by the HoloViews Pane.
    """
    clean_internal_figure_properties(fig_dict)
    config = fig_dict.pop('config', {})
    if config.get('responsive'):
        kwargs['sizing_mode'] = 'stretch_both'
    plotly_pane = pn.pane.Plotly(fig_dict, viewport_update_policy='mouseup', config=config, **kwargs)
    for callback_cls in callbacks.values():
        for callback_prop in callback_cls.callback_properties:
            plotly_pane.param.watch(lambda event, cls=callback_cls, prop=callback_prop: cls.update_streams_from_property_update(prop, event.new, event.obj.object), callback_prop)
    return plotly_pane