from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import network
from . import page
@event_class('Preload.preloadingAttemptSourcesUpdated')
@dataclass
class PreloadingAttemptSourcesUpdated:
    """
    Send a list of sources for all preloading attempts in a document.
    """
    loader_id: network.LoaderId
    preloading_attempt_sources: typing.List[PreloadingAttemptSource]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> PreloadingAttemptSourcesUpdated:
        return cls(loader_id=network.LoaderId.from_json(json['loaderId']), preloading_attempt_sources=[PreloadingAttemptSource.from_json(i) for i in json['preloadingAttemptSources']])