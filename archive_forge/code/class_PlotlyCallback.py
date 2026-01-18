from weakref import WeakValueDictionary
from ...element import Tiles
from ...streams import (
from .util import _trace_to_subplot
class PlotlyCallback(metaclass=PlotlyCallbackMetaClass):

    def __init__(self, plot, streams, source, **params):
        self.plot = plot
        self.streams = streams
        self.source = source
        self.last_event = None

    @classmethod
    def update_streams_from_property_update(cls, property, property_value, fig_dict):
        event_data = cls.get_event_data_from_property_update(property, property_value, fig_dict)
        streams = []
        for trace_uid, stream_data in event_data.items():
            if trace_uid in cls.instances:
                cb = cls.instances[trace_uid]
                try:
                    unchanged = stream_data == cb.last_event
                except Exception:
                    unchanged = False
                if unchanged:
                    continue
                cb.last_event = stream_data
                for stream in cb.streams:
                    stream.update(**stream_data)
                    streams.append(stream)
        try:
            Stream.trigger(streams)
        except Exception as e:
            raise e

    @classmethod
    def get_event_data_from_property_update(cls, property, property_value, fig_dict):
        raise NotImplementedError