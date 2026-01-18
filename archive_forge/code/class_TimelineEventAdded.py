from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import network
from . import page
@event_class('PerformanceTimeline.timelineEventAdded')
@dataclass
class TimelineEventAdded:
    """
    Sent when a performance timeline event is added. See reportPerformanceTimeline method.
    """
    event: TimelineEvent

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> TimelineEventAdded:
        return cls(event=TimelineEvent.from_json(json['event']))