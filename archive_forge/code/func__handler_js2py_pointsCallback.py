import ipywidgets as widgets
from traitlets import List, Unicode, Dict, observe, Integer
from .basedatatypes import BaseFigure, BasePlotlyType
from .callbacks import BoxSelector, LassoSelector, InputDeviceState, Points
from .serializers import custom_serializers
from .version import __frontend_version__
@observe('_js2py_pointsCallback')
def _handler_js2py_pointsCallback(self, change):
    """
        Process points callback message from the frontend
        """
    callback_data = change['new']
    if not callback_data:
        self._js2py_pointsCallback = None
        return
    event_type = callback_data['event_type']
    if callback_data.get('selector', None):
        selector_data = callback_data['selector']
        selector_type = selector_data['type']
        selector_state = selector_data['selector_state']
        if selector_type == 'box':
            selector = BoxSelector(**selector_state)
        elif selector_type == 'lasso':
            selector = LassoSelector(**selector_state)
        else:
            raise ValueError('Unsupported selector type: %s' % selector_type)
    else:
        selector = None
    if callback_data.get('device_state', None):
        device_state_data = callback_data['device_state']
        state = InputDeviceState(**device_state_data)
    else:
        state = None
    points_data = callback_data['points']
    trace_points = {trace_ind: {'point_inds': [], 'xs': [], 'ys': [], 'trace_name': self._data_objs[trace_ind].name, 'trace_index': trace_ind} for trace_ind in range(len(self._data_objs))}
    for x, y, point_ind, trace_ind in zip(points_data['xs'], points_data['ys'], points_data['point_indexes'], points_data['trace_indexes']):
        trace_dict = trace_points[trace_ind]
        trace_dict['xs'].append(x)
        trace_dict['ys'].append(y)
        trace_dict['point_inds'].append(point_ind)
    for trace_ind, trace_points_data in trace_points.items():
        points = Points(**trace_points_data)
        trace = self.data[trace_ind]
        if event_type == 'plotly_click':
            trace._dispatch_on_click(points, state)
        elif event_type == 'plotly_hover':
            trace._dispatch_on_hover(points, state)
        elif event_type == 'plotly_unhover':
            trace._dispatch_on_unhover(points, state)
        elif event_type == 'plotly_selected':
            trace._dispatch_on_selection(points, selector)
        elif event_type == 'plotly_deselect':
            trace._dispatch_on_deselect(points)
    self._js2py_pointsCallback = None