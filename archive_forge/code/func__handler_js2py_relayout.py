import ipywidgets as widgets
from traitlets import List, Unicode, Dict, observe, Integer
from .basedatatypes import BaseFigure, BasePlotlyType
from .callbacks import BoxSelector, LassoSelector, InputDeviceState, Points
from .serializers import custom_serializers
from .version import __frontend_version__
@observe('_js2py_relayout')
def _handler_js2py_relayout(self, change):
    """
        Process Plotly.relayout message from the frontend
        """
    relayout_msg = change['new']
    if not relayout_msg:
        self._js2py_relayout = None
        return
    relayout_data = relayout_msg['relayout_data']
    source_view_id = relayout_msg['source_view_id']
    if 'lastInputTime' in relayout_data:
        relayout_data.pop('lastInputTime')
    self.plotly_relayout(relayout_data=relayout_data, source_view_id=source_view_id)
    self._js2py_relayout = None