from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import runtime
@dataclass
class AnimationEffect:
    """
    AnimationEffect instance
    """
    delay: float
    end_delay: float
    iteration_start: float
    iterations: float
    duration: float
    direction: str
    fill: str
    easing: str
    backend_node_id: typing.Optional[dom.BackendNodeId] = None
    keyframes_rule: typing.Optional[KeyframesRule] = None

    def to_json(self):
        json = dict()
        json['delay'] = self.delay
        json['endDelay'] = self.end_delay
        json['iterationStart'] = self.iteration_start
        json['iterations'] = self.iterations
        json['duration'] = self.duration
        json['direction'] = self.direction
        json['fill'] = self.fill
        json['easing'] = self.easing
        if self.backend_node_id is not None:
            json['backendNodeId'] = self.backend_node_id.to_json()
        if self.keyframes_rule is not None:
            json['keyframesRule'] = self.keyframes_rule.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(delay=float(json['delay']), end_delay=float(json['endDelay']), iteration_start=float(json['iterationStart']), iterations=float(json['iterations']), duration=float(json['duration']), direction=str(json['direction']), fill=str(json['fill']), easing=str(json['easing']), backend_node_id=dom.BackendNodeId.from_json(json['backendNodeId']) if 'backendNodeId' in json else None, keyframes_rule=KeyframesRule.from_json(json['keyframesRule']) if 'keyframesRule' in json else None)