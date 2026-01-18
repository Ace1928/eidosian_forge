from weakref import WeakValueDictionary
from ...element import Tiles
from ...streams import (
from .util import _trace_to_subplot
class RangeCallback(PlotlyCallback):
    callback_properties = ['viewport', 'relayout_data']
    x_range = False
    y_range = False

    @classmethod
    def get_event_data_from_property_update(cls, property, property_value, fig_dict):
        traces = fig_dict.get('data', [])
        if property == 'viewport':
            event_data = cls.build_event_data_from_viewport(traces, property_value)
        else:
            event_data = cls.build_event_data_from_relayout_data(traces, property_value)
        return event_data

    @classmethod
    def build_event_data_from_viewport(cls, traces, property_value):
        event_data = {}
        for trace in traces:
            trace_type = trace.get('type', 'scatter')
            trace_uid = trace.get('uid', None)
            if _trace_to_subplot.get(trace_type, None) != ['xaxis', 'yaxis']:
                continue
            xaxis = trace.get('xaxis', 'x').replace('x', 'xaxis')
            yaxis = trace.get('yaxis', 'y').replace('y', 'yaxis')
            xprop = f'{xaxis}.range'
            yprop = f'{yaxis}.range'
            if not property_value:
                x_range = None
                y_range = None
            elif xprop in property_value and yprop in property_value:
                x_range = tuple(property_value[xprop])
                y_range = tuple(property_value[yprop])
            elif xprop + '[0]' in property_value and xprop + '[1]' in property_value and (yprop + '[0]' in property_value) and (yprop + '[1]' in property_value):
                x_range = (property_value[xprop + '[0]'], property_value[xprop + '[1]'])
                y_range = (property_value[yprop + '[0]'], property_value[yprop + '[1]'])
            else:
                continue
            stream_data = {}
            if cls.x_range:
                stream_data['x_range'] = x_range
            if cls.y_range:
                stream_data['y_range'] = y_range
            event_data[trace_uid] = stream_data
        return event_data

    @classmethod
    def build_event_data_from_relayout_data(cls, traces, property_value):
        event_data = {}
        for trace in traces:
            trace_type = trace.get('type', 'scattermapbox')
            trace_uid = trace.get('uid', None)
            if _trace_to_subplot.get(trace_type, None) != ['mapbox']:
                continue
            subplot_id = trace.get('subplot', 'mapbox')
            derived_prop = subplot_id + '._derived'
            if not property_value:
                x_range = None
                y_range = None
            elif 'coordinates' in property_value.get(derived_prop, {}):
                coords = property_value[derived_prop]['coordinates']
                (lon_top_left, lat_top_left), (lon_top_right, lat_top_right), (lon_bottom_right, lat_bottom_right), (lon_bottom_left, lat_bottom_left) = coords
                lon_left = min(lon_top_left, lon_bottom_left)
                lon_right = max(lon_top_right, lon_bottom_right)
                lat_bottom = min(lat_bottom_left, lat_bottom_right)
                lat_top = max(lat_top_left, lat_top_right)
                x_range, y_range = Tiles.lon_lat_to_easting_northing([lon_left, lon_right], [lat_bottom, lat_top])
                x_range = tuple(x_range)
                y_range = tuple(y_range)
            else:
                continue
            stream_data = {}
            if cls.x_range:
                stream_data['x_range'] = x_range
            if cls.y_range:
                stream_data['y_range'] = y_range
            event_data[trace_uid] = stream_data
        return event_data