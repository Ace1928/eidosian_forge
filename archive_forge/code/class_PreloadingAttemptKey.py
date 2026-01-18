from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import network
from . import page
@dataclass
class PreloadingAttemptKey:
    """
    A key that identifies a preloading attempt.

    The url used is the url specified by the trigger (i.e. the initial URL), and
    not the final url that is navigated to. For example, prerendering allows
    same-origin main frame navigations during the attempt, but the attempt is
    still keyed with the initial URL.
    """
    loader_id: network.LoaderId
    action: SpeculationAction
    url: str
    target_hint: typing.Optional[SpeculationTargetHint] = None

    def to_json(self):
        json = dict()
        json['loaderId'] = self.loader_id.to_json()
        json['action'] = self.action.to_json()
        json['url'] = self.url
        if self.target_hint is not None:
            json['targetHint'] = self.target_hint.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(loader_id=network.LoaderId.from_json(json['loaderId']), action=SpeculationAction.from_json(json['action']), url=str(json['url']), target_hint=SpeculationTargetHint.from_json(json['targetHint']) if 'targetHint' in json else None)