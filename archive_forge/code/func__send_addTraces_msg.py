import ipywidgets as widgets
from traitlets import List, Unicode, Dict, observe, Integer
from .basedatatypes import BaseFigure, BasePlotlyType
from .callbacks import BoxSelector, LassoSelector, InputDeviceState, Points
from .serializers import custom_serializers
from .version import __frontend_version__
def _send_addTraces_msg(self, new_traces_data):
    """
        Send Plotly.addTraces message to the frontend

        Parameters
        ----------
        new_traces_data : list[dict]
            List of trace data for new traces as accepted by Plotly.addTraces
        """
    layout_edit_id = self._last_layout_edit_id + 1
    self._last_layout_edit_id = layout_edit_id
    self._layout_edit_in_process = True
    trace_edit_id = self._last_trace_edit_id + 1
    self._last_trace_edit_id = trace_edit_id
    self._trace_edit_in_process = True
    add_traces_msg = {'trace_data': new_traces_data, 'trace_edit_id': trace_edit_id, 'layout_edit_id': layout_edit_id}
    self._py2js_addTraces = add_traces_msg
    self._py2js_addTraces = None