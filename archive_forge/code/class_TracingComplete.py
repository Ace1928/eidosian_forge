from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import io
@event_class('Tracing.tracingComplete')
@dataclass
class TracingComplete:
    """
    Signals that tracing is stopped and there is no trace buffers pending flush, all data were
    delivered via dataCollected events.
    """
    data_loss_occurred: bool
    stream: typing.Optional[io.StreamHandle]
    trace_format: typing.Optional[StreamFormat]
    stream_compression: typing.Optional[StreamCompression]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> TracingComplete:
        return cls(data_loss_occurred=bool(json['dataLossOccurred']), stream=io.StreamHandle.from_json(json['stream']) if 'stream' in json else None, trace_format=StreamFormat.from_json(json['traceFormat']) if 'traceFormat' in json else None, stream_compression=StreamCompression.from_json(json['streamCompression']) if 'streamCompression' in json else None)