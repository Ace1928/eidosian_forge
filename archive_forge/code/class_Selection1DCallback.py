from weakref import WeakValueDictionary
from ...element import Tiles
from ...streams import (
from .util import _trace_to_subplot
class Selection1DCallback(PlotlyCallback):
    callback_properties = ['selected_data']

    @classmethod
    def get_event_data_from_property_update(cls, property, selected_data, fig_dict):
        traces = fig_dict.get('data', [])
        point_inds = {}
        if selected_data:
            for point in selected_data['points']:
                point_inds.setdefault(point['curveNumber'], [])
                point_inds[point['curveNumber']].append(point['pointNumber'])
        event_data = {}
        for trace_ind, trace in enumerate(traces):
            trace_uid = trace.get('uid', None)
            new_index = point_inds.get(trace_ind, [])
            event_data[trace_uid] = dict(index=new_index)
        return event_data