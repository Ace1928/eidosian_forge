from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
@dataclass
class ContextRealtimeData:
    """
    Fields in AudioContext that change in real-time.
    """
    current_time: float
    render_capacity: float
    callback_interval_mean: float
    callback_interval_variance: float

    def to_json(self):
        json = dict()
        json['currentTime'] = self.current_time
        json['renderCapacity'] = self.render_capacity
        json['callbackIntervalMean'] = self.callback_interval_mean
        json['callbackIntervalVariance'] = self.callback_interval_variance
        return json

    @classmethod
    def from_json(cls, json):
        return cls(current_time=float(json['currentTime']), render_capacity=float(json['renderCapacity']), callback_interval_mean=float(json['callbackIntervalMean']), callback_interval_variance=float(json['callbackIntervalVariance']))