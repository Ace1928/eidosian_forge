import ipywidgets as widgets
from traitlets import List, Unicode, Dict, observe, Integer
from .basedatatypes import BaseFigure, BasePlotlyType
from .callbacks import BoxSelector, LassoSelector, InputDeviceState, Points
from .serializers import custom_serializers
from .version import __frontend_version__
@observe('_js2py_update')
def _handler_js2py_update(self, change):
    """
        Process Plotly.update message from the frontend
        """
    update_msg = change['new']
    if not update_msg:
        self._js2py_update = None
        return
    style = update_msg['style_data']
    trace_indexes = update_msg['style_traces']
    layout = update_msg['layout_data']
    source_view_id = update_msg['source_view_id']
    self.plotly_update(restyle_data=style, relayout_data=layout, trace_indexes=trace_indexes, source_view_id=source_view_id)
    self._js2py_update = None